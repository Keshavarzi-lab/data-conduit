'''
Infer the pixel positions of the nosepokes, per session, from pose at poke time.
================================================================================

Description:
    The rewarded-nosepoke analyses need each port's location in the camera image,
    but the rig does not record it. We infer it from pose: at the instant the animal
    pokes a port, its head sits at that port, so the head position at poke time is a
    proxy for the port's location.

    This is done PER SESSION because the camera may shift between sessions, so a
    port's pixel position is only stable within one session.

    Method (per session):
      1. For every completed trial that ended in an actual poke (a Success or a Fail;
         Misses have no poke), take the head position at the poke time. The head point
         is the nose by default, or the centroid of nose + both ears when
         ``use_ear_centroid=True`` (steadier, slightly behind the nose tip).
      2. Group those points by the poked port (``chosen_port``) and take the median,
         giving an inferred (x, y) per port that was poked.
      3. Numbering: on a Success the poked port equals the rewarded port, so the port
         identity is certain. We therefore mark ports that have at least one Success as
         "success-anchored", and report how far the Success-only median sits from the
         all-trials median, so a reader can judge how trustworthy each port's number is.

    An accuracy report accompanies the positions: how tightly the contributing points
    cluster (spread and maximum deviation), so we know how reliable each inferred
    position is. ``inter_port_distances`` then reports how far the inferred ports sit
    from one another (the ring layout should be roughly equidistant between neighbours).

    Nothing here calls ``movement``: it is xarray selection plus pandas aggregation,
    preparing coordinates the notebook can then use with ``movement``.

Contents:
--------------------------------
- infer_poke_positions:  Per-session inferred port positions + an accuracy report.
- inter_port_distances:  Pairwise (and nearest-neighbour) distances between ports.
'''

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# The three head keypoints used when building the ear-centroid head point.
_HEAD_KEYPOINTS = ('nose', 'lear', 'rear')


def _head_point(position: xr.DataArray, use_ear_centroid: bool) -> xr.DataArray:
    '''
    Reduce a pose position array to a single head point per frame.

    ----------
    Parameters:
        position (xr.DataArray):
            Pose positions with a ``keypoints`` dim and a ``space`` dim. May also
            carry a singleton ``individuals`` dim (movement schema) and a time dim
            named either ``'time'`` or ``'Time'``; both are handled.
        use_ear_centroid (bool):
            If True, the head point is the mean of nose, left ear and right ear. If
            False, it is the nose alone.
    Returns:
        xr.DataArray:
            Positions with dims ``(<time>, space)``: one head point per frame.
    '''
    # Drop the singleton individual dim if present (movement datasets carry it). Accept
    # either the singular ('individual', movement 0.17) or plural ('individuals') name.
    for individual_dim in ('individual', 'individuals'):
        if individual_dim in position.dims:
            position = position.isel({individual_dim: 0}, drop=True)
            break

    # The keypoint dimension is 'keypoint' in movement 0.17, 'keypoints' in older schemas.
    keypoint_dim = 'keypoint' if 'keypoint' in position.dims else 'keypoints'

    # Either the nose alone, or the centroid of nose + both ears.
    if use_ear_centroid:
        return position.sel({keypoint_dim: list(_HEAD_KEYPOINTS)}).mean(keypoint_dim)
    return position.sel({keypoint_dim: 'nose'}, drop=True)


def _time_name(array: xr.DataArray) -> str:
    '''
    Return the name of the time dimension, accepting movement's ``'time'`` or
    data-conduit's ``'Time'``.

    ----------
    Parameters:
        array (xr.DataArray):
            Any array carrying a time dimension.
    Returns:
        str:
            ``'time'`` or ``'Time'``.
    Raises:
        KeyError:
            If neither name is present.
    '''
    for candidate in ('time', 'Time'):
        if candidate in array.dims:
            return candidate
    raise KeyError("position has neither a 'time' nor a 'Time' dimension.")


def infer_poke_positions(
        position: xr.DataArray,
        trials: pd.DataFrame,
        *,
        use_ear_centroid: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Infer one session's nosepoke pixel positions and report their reliability.

    ----------
    Parameters:
        position (xr.DataArray):
            This session's pose positions (movement or data-conduit schema), on the
            same clock as ``trials`` poke times.
        trials (pd.DataFrame):
            This session's trials (from ``parse_trials``). Must have ``poke_time``,
            ``chosen_port``, ``correct_port`` and ``outcome``.
        use_ear_centroid (bool):
            Head point definition, forwarded to ``_head_point``. Default False (nose).
    Returns:
        (positions, accuracy):
            positions (pd.DataFrame), indexed by ``port`` (0-17, only poked ports):
              * ``x``, ``y``            inferred port position (median, pixels)
              * ``n_success``, ``n_fail``, ``n_total``  contributing poke counts
              * ``success_anchored``    True if >=1 Success poked this port
            accuracy (pd.DataFrame), indexed by ``port``:
              * ``spread_px``           mean distance of contributing points to the median
              * ``max_dev_px``          maximum such distance
              * ``std_x``, ``std_y``    coordinate standard deviations
              * ``success_vs_all_px``   distance between the Success-only median and the
                                        all-trials median (NaN if no Success), a check on
                                        how much fails pull the inferred position.
    '''
    # 1| Reduce pose to one head point per frame and find that point at each poke.
    head = _head_point(position, use_ear_centroid)
    tname = _time_name(head)

    # Keep only trials that ended in an actual poke (exclude Misses, chosen_port == -1).
    poked = trials[trials['chosen_port'] >= 0]
    if poked.empty:
        empty = pd.DataFrame(columns=['x', 'y', 'n_success', 'n_fail', 'n_total', 'success_anchored'])
        empty.index.name = 'port'
        acc = pd.DataFrame(columns=['spread_px', 'max_dev_px', 'std_x', 'std_y', 'success_vs_all_px'])
        acc.index.name = 'port'
        return empty, acc

    # 2| Look up the head point at each poke time (nearest frame), vectorised.
    poke_times = xr.DataArray(poked['poke_time'].to_numpy(), dims='poke')
    at_poke = head.sel({tname: poke_times}, method='nearest')

    # 3| Build one observation per poke: which port, which outcome, and where.
    obs = pd.DataFrame({
        'port': poked['chosen_port'].to_numpy(),
        'outcome': poked['outcome'].to_numpy(),
        'x': at_poke.sel(space='x').to_numpy(),
        'y': at_poke.sel(space='y').to_numpy(),
    })

    # 4| Inferred position per port = median of its observations.
    positions = obs.groupby('port')[['x', 'y']].median()

    # 5| Contributing counts per port (successes, fails, total), reindexed to the ports.
    n_success = obs[obs['outcome'] == 'success'].groupby('port').size()
    n_fail = obs[obs['outcome'] == 'fail'].groupby('port').size()
    positions['n_success'] = n_success.reindex(positions.index, fill_value=0).astype(int)
    positions['n_fail'] = n_fail.reindex(positions.index, fill_value=0).astype(int)
    positions['n_total'] = obs.groupby('port').size()
    # A port is trustworthy-numbered if a rewarded (Success) poke ever landed on it.
    positions['success_anchored'] = positions['n_success'] > 0

    # 6| Accuracy: spread of each port's contributing points around its median.
    merged = obs.merge(positions[['x', 'y']], on='port', suffixes=('', '_med'))
    merged['dev'] = np.hypot(merged['x'] - merged['x_med'], merged['y'] - merged['y_med'])
    accuracy = pd.DataFrame({
        'spread_px': merged.groupby('port')['dev'].mean(),
        'max_dev_px': merged.groupby('port')['dev'].max(),
        'std_x': obs.groupby('port')['x'].std(),
        'std_y': obs.groupby('port')['y'].std(),
    })

    # 7| Success-only median vs all-trials median: how much do fails shift the position?
    success_obs = obs[obs['outcome'] == 'success']
    if not success_obs.empty:
        succ_med = success_obs.groupby('port')[['x', 'y']].median()
        joined = succ_med.join(positions[['x', 'y']], lsuffix='_s', rsuffix='_a', how='right')
        accuracy['success_vs_all_px'] = np.hypot(joined['x_s'] - joined['x_a'], joined['y_s'] - joined['y_a'])
    else:
        accuracy['success_vs_all_px'] = np.nan

    return positions, accuracy


def inter_port_distances(positions: pd.DataFrame) -> pd.DataFrame:
    '''
    Pairwise distances between inferred port positions, with nearest-neighbour spacing.

    On the physical ring the ports are evenly spaced, so neighbouring inferred ports
    should sit at a roughly constant distance. This returns the full pairwise distance
    matrix (as a long table) plus, per port, the distance to its closest other port.

    ----------
    Parameters:
        positions (pd.DataFrame):
            Inferred positions indexed by ``port`` with ``x`` and ``y`` columns (the
            first return value of ``infer_poke_positions``).
    Returns:
        pd.DataFrame:
            Long-form with columns ``port_a``, ``port_b``, ``distance_px`` for every
            ordered pair (a != b), and a ``nearest`` boolean flagging, for each
            ``port_a``, the row holding its closest other port. Empty if fewer than two
            ports are present.
    '''
    # Need at least two ports to have any distance.
    ports = list(positions.index)
    if len(ports) < 2:
        return pd.DataFrame(columns=['port_a', 'port_b', 'distance_px', 'nearest'])

    # Compute every ordered pair's Euclidean distance.
    coords = positions[['x', 'y']]
    rows = []
    for a in ports:
        ax, ay = coords.loc[a, 'x'], coords.loc[a, 'y']
        for b in ports:
            if a == b:
                continue
            bx, by = coords.loc[b, 'x'], coords.loc[b, 'y']
            rows.append({'port_a': a, 'port_b': b, 'distance_px': float(np.hypot(ax - bx, ay - by))})
    table = pd.DataFrame(rows)

    # Flag, for each source port, the row that is its nearest neighbour.
    nearest_idx = table.groupby('port_a')['distance_px'].idxmin()
    table['nearest'] = False
    table.loc[nearest_idx, 'nearest'] = True
    return table

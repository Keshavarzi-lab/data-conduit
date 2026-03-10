'''
Segment module core.
---------------------

Description:
	Generic segmentation utilities for time-indexed data.

	The module provides two families of helpers:
	1) run-length segmentation from boolean/state series
	2) event-window slicing around timestamps for pandas/xarray objects

	These functions are designed as lightweight primitives that can be reused
	across datasource, multisource, and analysis layers.
'''


################################################################################
# Imports
################################################################################

import numpy as np
import pandas as pd
import xarray as xr

################################################################################




################################################################################
# Boolean Run-Length Segmentation
################################################################################

def segment_boolean_series(
	series: pd.Series,
	*,
	min_duration: float = 0.0,
	drop_inactive: bool = False,
) -> pd.DataFrame:
	'''
	Build run-length state segments from a time-indexed boolean/int series.

	Parameters
	----------
	series : pd.Series
		Time-indexed input. Values are coerced to boolean state (0/1).
	min_duration : float
		Minimum segment duration. Segments shorter than this are dropped.
	drop_inactive : bool
		If True, keep only active segments (`State == 1`).

	Returns
	-------
	pd.DataFrame
		Segment table with columns `Start`, `End`, `Duration`, `State`.

	Raises
	------
	TypeError
		If `series` is not a pandas Series.
		If `series` has no index object representing time.
	'''
	if not isinstance(series, pd.Series):
		raise TypeError('series must be a pandas Series.')
	if len(series) == 0:
		return pd.DataFrame(columns=['Start', 'End', 'Duration', 'State'])

	if not isinstance(series.index, pd.Index):
		raise TypeError('series must have an index representing time.')

	t = series.index.to_numpy(dtype=float)
	state = series.astype(bool).astype(int).to_numpy()

	transitions = np.flatnonzero(np.diff(state) != 0) + 1
	starts = np.r_[0, transitions]
	ends = np.r_[transitions, len(state)]

	out = []
	for s, e in zip(starts, ends, strict=True):
		start_t = float(t[s])
		end_t = float(t[e - 1])
		out.append(
			{
				'Start': start_t,
				'End': end_t,
				'Duration': end_t - start_t,
				'State': int(state[s]),
			}
		)

	segments = pd.DataFrame(out)
	if min_duration > 0:
		segments = segments.loc[segments['Duration'] >= min_duration]
	if drop_inactive:
		segments = segments.loc[segments['State'] == 1]

	return segments.reset_index(drop=True)


################################################################################




################################################################################
# Event-Window Slicing
################################################################################

def slice_event_windows(
	df: pd.DataFrame,
	event_times: list[float] | np.ndarray | pd.Series,
	*,
	pre: float,
	post: float,
) -> dict[float, pd.DataFrame]:
	'''
	Slice a DataFrame into event-centered windows.

	For each event time `ev`, a window is built from `[ev - pre, ev + post]`.
	The returned DataFrame index is shifted to event-relative time, so the event
	occurs at index 0.

	Parameters
	----------
	df : pd.DataFrame
		Input DataFrame with a numeric time index.
	event_times : array-like
		Event timestamps around which windows are extracted.
	pre : float
		Window length before each event.
	post : float
		Window length after each event.

	Returns
	-------
	dict[float, pd.DataFrame]
		Dictionary keyed by original event timestamp.
		Each value is the sliced DataFrame for that event.

	Raises
	------
	TypeError
		If `df` is not a pandas DataFrame.
	ValueError
		If `pre` or `post` is negative.
	'''
	if not isinstance(df, pd.DataFrame):
		raise TypeError('df must be a pandas DataFrame.')

	if pre < 0 or post < 0:
		raise ValueError('pre and post must be non-negative.')

	if len(df.index) == 0:
		return {}

	t = df.index.to_numpy(dtype=float)
	events = np.asarray(event_times, dtype=float)

	windows: dict[float, pd.DataFrame] = {}
	for ev in events:
		mask = (t >= (ev - pre)) & (t <= (ev + post))
		win = df.loc[mask].copy()
		if not win.empty:
			win.index = win.index.astype(float) - ev
		windows[float(ev)] = win

	return windows


def slice_dataarray_windows(
	da: xr.DataArray,
	event_times: list[float] | np.ndarray | pd.Series,
	*,
	pre: float,
	post: float,
	time_dim: str = 'Time',
) -> dict[float, xr.DataArray]:
	'''
	Slice an xarray DataArray into event-centered windows along `time_dim`.

	For each event time `ev`, a slice is extracted from `[ev - pre, ev + post]`.
	The selected coordinate values are then shifted so that `ev` maps to 0.

	Parameters
	----------
	da : xr.DataArray
		Input DataArray containing a time dimension.
	event_times : array-like
		Event timestamps around which windows are extracted.
	pre : float
		Window length before each event.
	post : float
		Window length after each event.
	time_dim : str
		Name of the time dimension to slice (default: `Time`).

	Returns
	-------
	dict[float, xr.DataArray]
		Dictionary keyed by original event timestamp.
		Each value is a sliced DataArray for that event.

	Raises
	------
	TypeError
		If `da` is not an xarray DataArray.
	ValueError
		If `time_dim` is missing or `pre`/`post` is negative.
	'''
	if not isinstance(da, xr.DataArray):
		raise TypeError('da must be an xarray DataArray.')
	if time_dim not in da.dims:
		raise ValueError(f"time_dim '{time_dim}' not found in DataArray dims: {da.dims}")
	if pre < 0 or post < 0:
		raise ValueError('pre and post must be non-negative.')

	times = da.coords[time_dim].values.astype(float)
	events = np.asarray(event_times, dtype=float)

	windows: dict[float, xr.DataArray] = {}
	for ev in events:
		mask = (times >= (ev - pre)) & (times <= (ev + post))
		sel_times = times[mask]
		win = da.sel({time_dim: sel_times})
		if win.sizes.get(time_dim, 0) > 0:
			rel_times = win.coords[time_dim].values.astype(float) - ev
			win = win.assign_coords({time_dim: rel_times})
		windows[float(ev)] = win

	return windows


################################################################################


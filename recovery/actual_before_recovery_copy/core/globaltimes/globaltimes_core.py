"""Create global time vectors and map stream timestamps onto them."""

import numpy as np
import pandas as pd
import xarray as xr


def _match_idx(
    base_array: np.ndarray,
    target_array: np.ndarray,
    *,
    match_type: str = "nearest",
) -> np.ndarray:
    """Return target indices matching each value in ``base_array``.

    Supported strategies are ``nearest``, ``before``, ``after``, and ``exact``.
    A value with no valid match is represented by ``-1``.
    """
    match_type = match_type.lower()
    if match_type not in {"nearest", "before", "after", "exact"}:
        raise ValueError("match_type must be one of: 'nearest', 'before', 'after', 'exact'.")

    if len(target_array) == 0:
        return np.full(len(base_array), -1, dtype=int)

    idx = np.searchsorted(target_array, base_array, side="left")

    if match_type == "exact":
        out = np.full(len(base_array), -1, dtype=int)
        in_bounds = idx < len(target_array)
        valid_positions = np.flatnonzero(in_bounds)
        exact_positions = valid_positions[target_array[idx[valid_positions]] == base_array[valid_positions]]
        out[exact_positions] = idx[exact_positions]
        return out

    if match_type == "before":
        out = np.searchsorted(target_array, base_array, side="right") - 1
        out[out < 0] = -1
        return out

    if match_type == "after":
        out = idx.copy()
        out[out >= len(target_array)] = -1
        return out

    left = np.clip(idx - 1, 0, len(target_array) - 1)
    right = np.clip(idx, 0, len(target_array) - 1)
    left_diff = np.abs(base_array - target_array[left])
    right_diff = np.abs(base_array - target_array[right])
    out = np.where((idx > 0) & (right_diff < left_diff), right, left)
    return out.astype(int)


def create_global_clock(
    start_time: float | int | None,
    end_time: float | int,
    timestep_interval: float | int,
    *,
    include_end_time: bool = True,
) -> np.ndarray:
    """Create a one-dimensional global clock at a fixed interval.

    ``None`` starts the clock at zero. When requested, ``end_time`` is appended
    if the interval-generated vector has not reached it.
    """
    if start_time is None:
        start_time = 0.0

    start = float(start_time)
    end = float(end_time)
    step = float(timestep_interval)

    if step == 0:
        raise ValueError("timestep_interval must be non-zero.")

    if start == end:
        return np.array([start], dtype=float) if include_end_time else np.array([], dtype=float)

    if (end > start and step < 0) or (end < start and step > 0):
        raise ValueError("timestep_interval must point from start_time toward end_time.")

    times = np.arange(start, end, step)
    if times.size == 0:
        return np.array([start, end], dtype=float) if include_end_time else np.array([], dtype=float)

    if include_end_time:
        last = float(times[-1])
        if (step > 0 and last < end) or (step < 0 and last > end):
            times = np.append(times, end)

    return times.astype(float)


def index_map_util(
    global_clock_times: np.ndarray,
    stream_times: np.ndarray | pd.DataFrame | xr.DataArray | pd.Series,
    *,
    match_type: str = "nearest",
) -> dict[str, pd.DataFrame | pd.Series]:
    """Map stream timestamps to each point on a global clock.

    The result contains compact index/time views, a delta series, and a ``full``
    table combining all of them. Missing matches use index ``-1`` and time/delta
    ``NaN``.
    """
    global_times = np.asarray(global_clock_times, dtype=float)
    if global_times.ndim != 1:
        raise ValueError("global_clock_times must be 1D.")

    if isinstance(stream_times, pd.DataFrame):
        if stream_times.shape[1] != 1:
            raise ValueError("stream_times DataFrame must have exactly one column.")
        times = stream_times.iloc[:, 0].to_numpy(dtype=float)
    elif isinstance(stream_times, pd.Series):
        times = stream_times.to_numpy(dtype=float)
    elif isinstance(stream_times, xr.DataArray):
        times = np.asarray(stream_times.values, dtype=float).ravel()
    else:
        times = np.asarray(stream_times, dtype=float).ravel()

    global_indices = np.arange(len(global_times))
    if times.size == 0:
        missing_indices = np.full(len(global_times), -1)
        missing_times = np.full(len(global_times), np.nan)
        return {
            "index_position_array": pd.DataFrame({"global_index": global_indices, "stream_index": missing_indices}),
            "matched_global_times": pd.DataFrame({"global_time": global_times, "stream_time": missing_times}),
            "global_stream_time_delta": pd.Series(missing_times, name="delta"),
            "full": pd.DataFrame(
                {
                    "global_index": global_indices,
                    "stream_index": missing_indices.astype(float),
                    "global_time": global_times,
                    "stream_time": missing_times,
                    "delta": missing_times,
                }
            ),
        }

    if times.size > 1 and not np.all(times[:-1] <= times[1:]):
        raise ValueError("stream_times must be monotonically non-decreasing.")

    stream_indices = _match_idx(global_times, times, match_type=match_type)
    matched_times = np.where(
        stream_indices >= 0,
        times[np.clip(stream_indices, 0, len(times) - 1)],
        np.nan,
    )
    delta = global_times - matched_times

    return {
        "index_position_array": pd.DataFrame({"global_index": global_indices, "stream_index": stream_indices}),
        "matched_global_times": pd.DataFrame({"global_time": global_times, "stream_time": matched_times}),
        "global_stream_time_delta": pd.Series(delta, name="delta"),
        "full": pd.DataFrame(
            {
                "global_index": global_indices,
                "stream_index": stream_indices.astype(float),
                "global_time": global_times,
                "stream_time": matched_times,
                "delta": delta,
            }
        ),
    }

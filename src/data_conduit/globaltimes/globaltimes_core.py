'''
Global timebase utilities.
--------------------------

Description:
	Utilities for creating a canonical/global time vector and mapping stream
	timestamps to that vector with configurable matching rules.
'''


################################################################################
# Imports
################################################################################

import numpy as np
import pandas as pd
import xarray as xr

################################################################################




################################################################################
# Private Helper
################################################################################

def _match_idx(
	base_array: np.ndarray,
	target_array: np.ndarray,
	*,
	match_type: str = 'nearest',
) -> np.ndarray:
	'''
	Return target indices that best match each value in base_array.
	'''
	match_type = match_type.lower()
	if match_type not in {'nearest', 'before', 'after', 'exact'}:
		raise ValueError("match_type must be one of: 'nearest', 'before', 'after', 'exact'.")

	idx = np.searchsorted(target_array, base_array, side='left')

	if match_type == 'exact':
		out = np.full(len(base_array), -1, dtype=int)
		ok = (idx < len(target_array)) & (target_array[idx] == base_array)
		out[ok] = idx[ok]
		return out

	if match_type == 'before':
		out = np.searchsorted(target_array, base_array, side='right') - 1
		out[out < 0] = -1
		return out

	if match_type == 'after':
		out = idx.copy()
		out[out >= len(target_array)] = -1
		return out

	# nearest
	left = np.clip(idx - 1, 0, len(target_array) - 1)
	right = np.clip(idx, 0, len(target_array) - 1)
	left_diff = np.abs(base_array - target_array[left])
	right_diff = np.abs(base_array - target_array[right])
	out = np.where((idx > 0) & (right_diff < left_diff), right, left)

	if len(target_array) == 0:
		return np.full(len(base_array), -1, dtype=int)
	return out.astype(int)


################################################################################




################################################################################
# Public API
################################################################################

def create_global_clock(
	start_time: float | int | None,
	end_time: float | int,
	timestep_interval: float | int,
	*,
	include_end_time: bool = True,
) -> np.ndarray:
	'''
	Create an evenly spaced global clock.
	'''
	if start_time is None:
		start_time = 0.0

	start = float(start_time)
	end = float(end_time)
	step = float(timestep_interval)

	if step == 0:
		raise ValueError('timestep_interval must be non-zero.')

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
	match_type: str = 'nearest',
) -> dict[str, np.ndarray]:
	'''
	Map stream timestamps to global clock timestamps.

	Returns
	-------
	dict
		{
		  'index_position_array': [N x 2] -> [global_index, stream_index],
		  'matched_global_times': [N x 2] -> [global_time, matched_stream_time],
		  'global_stream_time_delta': [N] -> (global_time - matched_stream_time)
		}
	'''
	g = np.asarray(global_clock_times, dtype=float)
	if g.ndim != 1:
		raise ValueError('global_clock_times must be 1D.')

	if isinstance(stream_times, pd.DataFrame):
		if stream_times.shape[1] != 1:
			raise ValueError('stream_times DataFrame must have exactly one column.')
		s = stream_times.iloc[:, 0].to_numpy(dtype=float)
	elif isinstance(stream_times, pd.Series):
		s = stream_times.to_numpy(dtype=float)
	elif isinstance(stream_times, xr.DataArray):
		s = np.asarray(stream_times.values, dtype=float).ravel()
	else:
		s = np.asarray(stream_times, dtype=float).ravel()

	if s.size == 0:
		missing = np.full(len(g), np.nan)
		return {
			'index_position_array': np.column_stack((np.arange(len(g)), np.full(len(g), -1))),
			'matched_global_times': np.column_stack((g, missing)),
			'global_stream_time_delta': np.full(len(g), np.nan),
		}

	stream_idx = _match_idx(g, s, match_type=match_type)
	matched = np.where(stream_idx >= 0, s[np.clip(stream_idx, 0, len(s) - 1)], np.nan)

	index_position_array = np.column_stack((np.arange(len(g)), stream_idx))
	matched_global_times = np.column_stack((g, matched))
	delta = g - matched

	return {
		'index_position_array': index_position_array,
		'matched_global_times': matched_global_times,
		'global_stream_time_delta': delta,
	}


################################################################################


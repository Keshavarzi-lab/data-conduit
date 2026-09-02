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

import warnings

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

	Parameters
	----------
	base_array : array-like
		1D array of values to match.
	target_array : array-like
		1D array of values to search within.
	match_type : {'nearest', 'before', 'after', 'exact'}, optional
		Matching rule for finding indices. Default is 'nearest'.
		- 'nearest': Return the index of the closest value in target_array.
		- 'before': Return the index of the largest value in target_array that is <= base_array.
		- 'after': Return the index of the smallest value in target_array that is >= base_array.
		- 'exact': Return the index of the value in target_array that is exactly equal to base_array, or -1 if no exact match exists.
	Returns
	-------
	ndarray
		Array of indices in target_array corresponding to each value in base_array according to the specified match_type. If no valid match is found for a given value, the index will be -1.
	'''
	match_type = match_type.lower()
	if match_type not in {'nearest', 'before', 'after', 'exact'}:
		raise ValueError("match_type must be one of: 'nearest', 'before', 'after', 'exact'.")

	if len(target_array) == 0:
		return np.full(len(base_array), -1, dtype=int)

	idx = np.searchsorted(target_array, base_array, side='left')

	if match_type == 'exact':
		out = np.full(len(base_array), -1, dtype=int)
		in_bounds = idx < len(target_array)
		valid_positions = np.flatnonzero(in_bounds)
		exact_positions = valid_positions[target_array[idx[valid_positions]] == base_array[valid_positions]]
		out[exact_positions] = idx[exact_positions]
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
	Create a global clock as a 1D array of timestamps from start_time to end_time with specified intervals.

	Parameters
	----------
	start_time : float or int or None
		Start time of the global clock. If None, defaults to 0.0.
	end_time : float or int
		End time of the global clock.
	timestep_interval : float or int
		Time interval between consecutive timestamps in the global clock. Must be non-zero.
	include_end_time : bool, optional
		If True, include end_time in the global clock if it falls on a valid timestep. Default is True.
	
	Returns
	-------
	ndarray
		1D array of timestamps representing the global clock, starting from start_time up to end_time with
		intervals of timestep_interval. If include_end_time is True and end_time does not fall on a valid 
		timestep, end_time will be included as the last timestamp in the array.	
	'''
	if start_time is None:
		start_time = 0.0

	start = float(start_time)
	end = float(end_time)
	step = float(timestep_interval)

	if step == 0:
		raise ValueError('timestep_interval must be non-zero.')

	if start == end:
		return np.array([start], dtype=float) if include_end_time else np.array([], dtype=float)

	if (end > start and step < 0) or (end < start and step > 0):
		raise ValueError('timestep_interval must point from start_time toward end_time.')

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
) -> dict[str, pd.DataFrame | pd.Series]:
	'''
	Map stream timestamps to global clock timestamps.

	Parameters
	----------
	global_clock_times : array-like
		1D array of global clock timestamps.
	stream_times : array-like or pd.DataFrame or xr.DataArray
		1D array or single-column DataFrame or 1D DataArray of stream timestamps.
	match_type : {'nearest', 'before', 'after', 'exact'}, optional
		Matching rule for mapping stream times to global times. Default is 'nearest'.

	Returns
	-------
	dict
		{
		  'index_position_array': DataFrame [N x 2] -> columns ['global_index', 'stream_index'],
		  'matched_global_times': DataFrame [N x 2] -> columns ['global_time', 'stream_time'],
		  'global_stream_time_delta': Series [N] -> (global_time - stream_time),
		  'full': DataFrame [N x 5] -> columns ['global_index', 'stream_index', 'global_time', 'stream_time', 'delta']
		}
			- 'index_position_array': DataFrame where each row contains the global index and the corresponding stream index (-1 if no match).
			- 'matched_global_times': DataFrame where each row contains the global time and the matched stream time (NaN if no match).
			- 'global_stream_time_delta': Series containing the difference between global time and matched stream time (NaN if no match).
			- 'full': DataFrame where each row contains global_index, stream_index, global_time, stream_time, and delta.

	Warns
	-----
	UserWarning
		If stream timestamps are not monotonically non-decreasing. Matching then
		uses a stable time-sorted view and returns indices into the original input.
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
		global_idx = np.arange(len(g))
		missing_idx = np.full(len(g), -1)
		missing = np.full(len(g), np.nan)
		return {
			'index_position_array': pd.DataFrame({'global_index': global_idx, 'stream_index': missing_idx}),
			'matched_global_times': pd.DataFrame({'global_time': g, 'stream_time': missing}),
			'global_stream_time_delta': pd.Series(missing, name='delta'),
			'full': pd.DataFrame({'global_index': global_idx, 'stream_index': missing_idx.astype(float), 'global_time': g, 'stream_time': missing, 'delta': missing}),
		}

	if s.size > 1 and not np.all(s[:-1] <= s[1:]):
		warnings.warn(
			'stream_times are not monotonically non-decreasing; matching will use '
			'a stable time-sorted view and return indices into the original input. '
			'Clock resets or overlapping time ranges may make matches ambiguous.',
			UserWarning,
			stacklevel=2,
		)
		sorted_to_original = np.argsort(s, kind='stable')
		sorted_stream_idx = _match_idx(g, s[sorted_to_original], match_type=match_type)
		stream_idx = np.full_like(sorted_stream_idx, -1)
		has_match = sorted_stream_idx >= 0
		stream_idx[has_match] = sorted_to_original[sorted_stream_idx[has_match]]
	else:
		stream_idx = _match_idx(g, s, match_type=match_type)

	matched = np.where(stream_idx >= 0, s[np.clip(stream_idx, 0, len(s) - 1)], np.nan)

	global_idx = np.arange(len(g))
	delta = g - matched

	return {
		'index_position_array': pd.DataFrame({'global_index': global_idx, 'stream_index': stream_idx}),
		'matched_global_times': pd.DataFrame({'global_time': g, 'stream_time': matched}),
		'global_stream_time_delta': pd.Series(delta, name='delta'),
		'full': pd.DataFrame({'global_index': global_idx, 'stream_index': stream_idx.astype(float), 'global_time': g, 'stream_time': matched, 'delta': delta}),
	}


################################################################################

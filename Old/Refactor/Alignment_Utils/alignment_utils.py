'''
Utilities for aligning timestamps across devices and modalities.

Contains functions for:
- Global clock creation and index mapping
- Video timestamp extraction
- Digital and Neuropixel sync pulse extraction
- Sync pulse alignment and NPX-to-Bonsai time conversion
- Visualisation of sync pulses

Note:  Bonsai_DefaultSession (session orchestrator) is intentionally excluded
       from this module because it depends on many other data-loading classes
       (ExperimentEvents, RotationData, VideoData, BehaviorNosepoke, etc.).
       It should live in its own session module.
'''



########################################################################################################################
# Imports
########################################################################################################################

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import os


########################################################################################################################









########################################################################################################################
# Private Helper Functions
########################################################################################################################




#===============================================================================
# 1| _match_idx
#===============================================================================


def _match_idx(base_array= None,
               target_array= None,
               match_type= 'nearest',
               verbose= False):
    '''
    Matches target_array times to base_array times and returns index array of
    matched indices using searchsorted.

    Parameters
    ----------
    base_array : np.ndarray or convertible to 1D array
        1D array of base times to match against.
    target_array : np.ndarray or convertible to 1D array
        1D array of target times to find matches for.
    match_type : str
        Method for matching. Options are 'nearest', 'before', 'after', 'exact'.
    verbose : bool
        If True, prints information about the matching process.

    Returns
    -------
    matched_indices : np.ndarray
        1D array of indices into target_array that best match each element of
        base_array according to match_type.
    '''

    #==== Convert inputs to numpy arrays =====

    if not isinstance(base_array, np.ndarray):
        base_array= np.array(base_array)
        if verbose:
            print('Converted base_array to numpy array.')
    if not isinstance(target_array, np.ndarray):
        target_array= np.array(target_array)
        if verbose:
            print('Converted target_array to numpy array.')

    # Check if 1D, else assume first dimension is intended dimension
    if base_array.ndim != 1:
        base_array= base_array[:, 0]
        if verbose:
            print('base_array is not 1D. Using first dimension for matching.')
    if target_array.ndim != 1:
        target_array= target_array[:, 0]
        if verbose:
            print('target_array is not 1D. Using first dimension for matching.')

    #==== Match_type handling ================

    # ensure case insensitivity
    if not isinstance(match_type, str):
        raise ValueError('match_type must be a string.')

    match_type= match_type.lower()
    if match_type not in ['nearest', 'before', 'after', 'exact']:
        raise ValueError(f'Unsupported match_type: {match_type}. Supported types are: nearest, before, after, exact.')


    #==== Matching Logic =====================


    #==== 1) Exact Match =====================
    if match_type == 'exact':
        idx = np.searchsorted(target_array, base_array, side="left")
        exact = (idx < len(target_array)) & (target_array[idx] == base_array)
        out = np.full(len(base_array), 0, dtype=int)
        out[exact] = idx[exact]
        return out


    #==== 2) Previous Match ==================
    if match_type == 'before':
        idx = np.searchsorted(target_array, base_array, side="right") - 1
        idx[idx < 0] = 0
        return idx


    #==== 3) Next Match ======================
    if match_type == 'after':
        idx = np.searchsorted(target_array, base_array, side="left")
        idx[idx >= len(target_array)] = 0
        return idx


    #==== 4) Nearest Match ===================
    if match_type == 'nearest':
        idx = np.searchsorted(target_array, base_array, side="left")
        idx_left = np.clip(idx - 1, 0, len(target_array) - 1)
        idx_right = np.clip(idx, 0, len(target_array) - 1)

        left_diff = np.abs(base_array - target_array[idx_left])
        right_diff = np.abs(base_array - target_array[idx_right])

        nearest_idx = np.where((idx > 0) & (right_diff < left_diff), idx_right, idx_left)

        # Handle edge cases
        nearest_idx[idx == 0] = 0
        nearest_idx[idx == len(target_array)] = len(target_array) - 1

        return nearest_idx


#===============================================================================




########################################################################################################################
# Global Clock & Index Mapping
########################################################################################################################




#===============================================================================
# 2| create_global_clock
#===============================================================================


def create_global_clock(start_time= None,
                        end_time= None,
                        timestep_interval= 1,
                        include_end_time= True,
                        verbose= False):
    '''
    Creates a global clock time array from start_time to end_time with specified
    timestep_interval.

    Parameters
    ----------
    start_time : float or int or None
        Start time of the global clock. If None, defaults to 0.
    end_time : float or int
        End time of the global clock.
    timestep_interval : float or int
        Time interval between each timestep in the global clock.
    include_end_time : bool
        If True, includes end_time in the global clock array even if not an
        exact multiple of timestep_interval.
    verbose : bool
        If True, prints information about the created global clock.

    Returns
    -------
    global_times : np.ndarray
        Array of global clock times.
    '''
    #==== Handle start_time default =====
    if start_time is None:
        start_time = 0
        if verbose:
            print(f'Warning: start_time not provided. Defaulting to 0.')

    #==== Handle Errors =================
    if end_time is None:
        raise ValueError('end_time must be provided.')
    if timestep_interval <= 0 and end_time > start_time:
        raise ValueError('timestep_interval must be positive when end_time > start_time.')
    if timestep_interval >= 0 and end_time < start_time:
        raise ValueError('timestep_interval must be negative when end_time < start_time.')

    #==== Create global clock times =====
    times= np.arange(start_time, end_time, timestep_interval)

    #==== Include end_time if specified =
    if include_end_time is True:
        if (timestep_interval > 0 and times[-1] < end_time) or (timestep_interval < 0 and times[-1] > end_time):
            times= np.append(times, end_time)
            if verbose:
                print(f'Including end_time: {end_time} in global clock times. Final time array length: {len(times)}, with final timestep size: {times[-1] - times[-2]}')
    elif len(times) == 0:
        raise ValueError(f'No times within specified range. Check start_time, end_time, and timestep_interval values.')
    elif times[-1] != end_time and verbose:
        print(f'Note: end_time: {end_time} not included in global clock times. Final time: {times[-1]}, End time: {end_time}')

    if verbose:
        print(f'Created global clock times from {start_time} to {end_time} with timestep interval {timestep_interval}. Total timesteps: {len(times)}')
    return times


#===============================================================================




#===============================================================================
# 3| index_map_util
#===============================================================================


# def index_map_util(global_clock_times: np.ndarray,
#                    stream_times: np.ndarray | pd.DataFrame | xr.DataArray,
#                    match_type: str = 'Nearest',
#                    verbose: bool = False
#                    ):
#     '''
#     Utility function to create index mapping from stream_times to
#     global_clock_times based on match_type.

#     Parameters
#     ----------
#     global_clock_times : np.ndarray
#         Array of global clock times to match against.
#     stream_times : np.ndarray or pd.DataFrame or xr.DataArray
#         Array or DataFrame or DataArray of stream times to find index of nearest
#         matches for. Should be provided as column containing time values.
#     match_type : str
#         Type of matching to perform. Options are 'Nearest', 'Before', 'After',
#         'Exact'.
#     verbose : bool
#         If True, prints information about the index mapping process.

#     Returns
#     -------
#     times_matrix : np.ndarray
#         (N x 2) array: [global_time, matched_stream_time].
#     index_matrix : np.ndarray
#         (N x 2) array: [global_index, stream_index].
#     full_matrix : np.ndarray
#         (N x 5) array: [global_index, stream_index, global_time, stream_time,
#         time_difference].
#     '''

#     #==== Check global_clock_times is 1D array ===================================
#     if len(global_clock_times.shape) != 1:
#         raise ValueError('global_clock_times must be a 1D array.')

#     #==== Convert stream_times to np.ndarray if pd.DataFrame or xr.DataArray =====
#     if isinstance(stream_times, pd.DataFrame):
#         if stream_times.shape[1] != 1:
#             raise ValueError('If stream_times is a DataFrame, it must contain only one column of time values.')
#         stream_times_array= stream_times.iloc[:, 0].values
#     elif isinstance(stream_times, xr.DataArray):
#         stream_times_array= stream_times.values
#     else:
#         stream_times_array= stream_times

#     #==== Match stream_times to global_clock_times ===============================

#     matched_indices= _match_idx(base_array= global_clock_times,
#                                 target_array= stream_times_array,
#                                 match_type= match_type,
#                                 verbose= verbose)

#     # Retrieve matched stream-time values (use NaN for missing)
#     matched_stream_vals = np.where(matched_indices >= 0,
#                                    stream_times_array[matched_indices],
#                                    np.nan)

#     # A. (N x 2) array: [global_time, matched_stream_time]
#     times_matrix = np.column_stack((global_clock_times, matched_stream_vals))

#     # B. (N x 2) array: [global_index, stream_index]
#     global_idx = np.arange(len(global_clock_times))
#     index_matrix = np.column_stack((global_idx, matched_indices))

#     # C. (N x 5) combined array:
#     #    [global_index, stream_index, global_time, stream_time, time_difference]
#     time_difference = global_clock_times - matched_stream_vals
#     full_matrix = np.column_stack((
#         global_idx,
#         matched_indices,
#         global_clock_times,
#         matched_stream_vals,
#         time_difference
#     ))

#     return times_matrix, index_matrix, full_matrix


def index_map_util(global_clock_times: np.ndarray,
                   stream_times: np.ndarray | pd.DataFrame | xr.DataArray,
                   match_type: str = 'Nearest',
                   verbose: bool = False
                   ):
    '''
    Utility function to create index mapping from stream_times to global_clock_times based on match_type.
    Parameters
    ----------
    global_clock_times : np.ndarray
        Array of global clock times to match against.
    stream_times : np.ndarray or pd.DataFrame or xr.DataArray
        Array or DataFrame or DataArray of stream times to find index of nearest matches for. Should be provided as column containing time values.
    match_type : str
        Type of matching to perform. Options are 'Nearest', 'Before', 'After', 'Exact'.
    verbose : bool
        If True, prints information about the index mapping process.
    Returns
    -------
    index_map_dict: dict
        Dictionary containing:
            'index_position_array': 2D np.ndarray with index positions of matched times. First column: global clock index, Second column: stream time index
            'matched_global_times': 2D np.ndarray of matched global clock times. First column: global clock time, Second column: stream time
            'global_stream_time_delta': 1D np.ndarray of time differences between matched global clock times and stream times
    '''

    #==== Check global_clock_times is 1D array ===================================
    if len(global_clock_times.shape) != 1:
        raise ValueError('global_clock_times must be a 1D array.')
    
    #==== Convert stream_times to np.ndarray if pd.DataFrame or xr.DataArray =====
    if isinstance(stream_times, pd.DataFrame):
        if stream_times.shape[1] != 1:
            raise ValueError('If stream_times is a DataFrame, it must contain only one column of time values.')
        stream_times_array= stream_times.iloc[:,0].values
    elif isinstance(stream_times, xr.DataArray):
        stream_times_array= stream_times.values
    else:
        stream_times_array= stream_times
    
    #==== Create Dictionary for index mapping results ============================
    index_map_dict = {}

    #==== Match stream_times to global_clock_times ===============================

    matched_indices= _match_idx(base_array= global_clock_times,
                               target_array= stream_times_array,
                               match_type= match_type,
                               verbose= verbose)
    
    # 2. Retrieve matched stream-time values (use NaN for missing)
    matched_stream_vals = np.where(matched_indices >= 0,
                                   stream_times_array[matched_indices],
                                   np.nan)

    # A. (N×2) array: [global_time, matched_stream_time]
    times_matrix = np.column_stack((global_clock_times, matched_stream_vals))

    # B. (N×2) array: [global_index, stream_index]
    global_idx = np.arange(len(global_clock_times))
    index_matrix = np.column_stack((global_idx, matched_indices))

    # C. (N×5) combined array:
    # [global_index, stream_index, global_time, stream_time, time_difference]
    time_difference = global_clock_times - matched_stream_vals
    full_matrix = np.column_stack((
        global_idx,
        matched_indices,
        global_clock_times,
        matched_stream_vals,
        time_difference
    ))

    return times_matrix, index_matrix, full_matrix

#===============================================================================




########################################################################################################################
# Video Timestamps
########################################################################################################################




#===============================================================================
# 4| get_video_timestamps
#===============================================================================


def get_video_timestamps(experiment_directory_path= None,
                         video_path= None,
                         auto_match_device= True,
                         harp_device_yaml_path= './device.yml',
                         device_type= 'Behavior',
                         device_list= ['Behavior0', 'Behavior1', 'Behavior2',
                                       'Behavior3', 'Behavior4', 'Behavior5'],
                         device_IDs= {'Behavior0': 'ID_0', 'Behavior1': 'ID_1',
                                      'Behavior2': 'ID_2', 'Behavior3': 'ID_3',
                                      'Behavior4': 'ID_4', 'Behavior5': 'ID_5'},
                         device_registers_dict= {'Behavior0': ['92'], 'Behavior1': ['92'],
                                                 'Behavior2': ['92'], 'Behavior3': ['92'],
                                                 'Behavior4': ['92'], 'Behavior5': ['92']},
                         da_type_dict= {'Behavior0_Camera0Frame': ('Behavior0', '92'),
                                        'Behavior1_Camera0Frame': ('Behavior1', '92'),
                                        'Behavior2_Camera0Frame': ('Behavior2', '92'),
                                        'Behavior3_Camera0Frame': ('Behavior3', '92'),
                                        'Behavior4_Camera0Frame': ('Behavior4', '92'),
                                        'Behavior5_Camera0Frame': ('Behavior5', '92')},
                         matching_only= False,
                         verbose= False):
    '''
    Returns xr.DataArray with frame index and timestamps for video frames
    based on matching harp device data.

    Automatically matches the correct Camera0Frames device by comparing the
    number of video frames to the number of timestamps per device.

    Parameters
    ----------
    experiment_directory_path : str
        Path to experiment directory containing bonsai logs.
    video_path : str
        Path to video file.
    auto_match_device : bool
        If True, automatically matches device based on number of frames.
    harp_device_yaml_path : str
        Path to harp device YAML file.
    device_type : str
        Device type prefix for Camera0Frames.
    device_list : list of str
        List of device names to search.
    device_IDs : dict
        Dictionary mapping device names to their IDs.
    device_registers_dict : dict
        Dictionary of device names to register address lists.
    da_type_dict : dict
        Dictionary mapping output name to (device_name, register_address).
    matching_only : bool
        If True, only return matching device data.
    verbose : bool
        If True, prints information about matching.

    Returns
    -------
    frame_times_da : xr.DataArray
        DataArray with frame indices as coordinates and timestamps as values.
    '''
    from sleap_io import Video as _Video

    # Lazy import — Camera0Frames lives in the Devices module alongside this
    # refactored code.  Fall back to the legacy Helper_functions location if
    # the refactored module has not been installed yet.
    try:
        from Refactor.Devices.devices import Camera0Frames as _Camera0Frames
    except ImportError:
        from Data_Imports.Helper_Functions.Helper_functions import Camera0Frames as _Camera0Frames

    #==== Load video and get frame count ====
    video= _Video.from_filename(filename= video_path)
    frame_count= video.__len__()
    if verbose:
        print(f'Video frame count: {frame_count}')

    #==== Load frame times from harp device ====
    frame_time_dfs= _Camera0Frames(experiment_directory_path= experiment_directory_path,
                                    harp_device_yaml_path= harp_device_yaml_path,
                                    device_type= device_type,
                                    device_list= device_list,
                                    device_IDs= device_IDs,
                                    device_registers_dict= device_registers_dict,
                                    da_type_dict= da_type_dict,
                                    matching_only= matching_only,
                                    verbose= verbose
                                    ).device_da_dict
    if verbose:
        print(f'Available frame time devices: {list(frame_time_dfs.keys())}')
        if auto_match_device:
            print('Auto-matching device based on frame count...')

    #==== Auto-match device based on frame count ====
    if auto_match_device:
        matching_device= None

        for key in frame_time_dfs.keys():
            da = frame_time_dfs[key]
            if len(da['Time']) == frame_count:
                matching_device= key
                if verbose:
                    print(f'Matched device: {matching_device} with {len(da["Time"])} frames')
                break

        if matching_device is None:
            raise ValueError('No matching device found for video frame count')
        frame_time_df= frame_time_dfs[matching_device]
    else:
        frame_time_df= list(frame_time_dfs.values())[0]
        if verbose:
            print(f'Using first available device: {list(frame_time_dfs.keys())[0]} with {len(frame_time_df["Time"])} frames')

    #==== Create DataArray of frame indices and timestamps ====
    frame_times_da = xr.DataArray(
        data= frame_time_df['Time'].values,
        dims= ['frame'],
        coords= {'frame': np.arange(len(frame_time_df['Time']))},
        name= 'timestamp'
    )

    return frame_times_da


#===============================================================================




########################################################################################################################
# Sync Pulse Extraction
########################################################################################################################




#===============================================================================
# 5| get_square_waveform
#===============================================================================


def get_square_waveform(oneD_data= None,
                        start_times= None,
                        end_times= None,
                        resolution= 0.001,
                        start_val= None,
                        end_val= None,
                        conversion_ratio= 1,
                        min_separation= None,
                        decimal_places= 11,
                        align_to_zero= True,
                        df_active_only= True):
    '''
    Extracts square waveform from either 1D timeseries data or from explicit
    start and end times.

    Parameters
    ----------
    oneD_data : array-like or None
        1D timeseries data containing square waves. Not used with start_times /
        end_times.
    start_times : array-like or None
        Start times of square waves. Not used with oneD_data.
    end_times : array-like or None
        End times of square waves. Not used with oneD_data.
    resolution : float
        Time resolution when generating timeseries from start/end times.
    start_val : float or None
        Starting time offset for timeseries data. Defaults to 0.
    end_val : float or None
        End time used to trim the generated timeseries.
    conversion_ratio : float
        Conversion ratio applied to sample indices to obtain time values.
        Default is 1 (no conversion).
    min_separation : float or None
        Minimum separation between pulses. Segments shorter than this are
        merged with neighbours. Default is None (no minimum).
    decimal_places : int or None
        Number of decimal places to round time values to. Default is 11.
    align_to_zero : bool
        Whether to align the first pulse start to time zero. Default is True.
    df_active_only : bool
        If True, drops inactive (state == 0) segments from the returned
        DataFrame. Default is True.

    Returns
    -------
    times : np.ndarray
        Time values for the square waveform.
    time_offset : float
        Time offset of the first pulse start (before zero-alignment).
    data : np.ndarray
        Square waveform data (0 for low, 100 for high).
    pulse_durations : pd.DataFrame
        DataFrame with Start, End, Duration, and State columns for each segment.
    '''
    #==== Input validation and preparation ======================================
    if (oneD_data is None) == (start_times is None or end_times is None):
        raise ValueError("Provide either 'oneD_data' or both 'start_times' and 'end_times', but not both.")


    #==== Option 1: Extract square waveform from oneD_data ======================
    if oneD_data is not None:
        data= np.asarray(oneD_data)

        #=== Input validation ===
        if data.ndim != 1:
            raise ValueError("'oneD_data' must be a 1D array-like structure.")
        if conversion_ratio is None or conversion_ratio <= 0:
            raise ValueError("'conversion_ratio' must be a positive number.")
        if start_val is None:
            start_val = 0.0
        if end_val is not None and end_val < start_val:
            raise ValueError("'end_val' cannot be earlier than 'start_val'.")

        #=== Get time values ===
        times= start_val + np.arange(len(data)) * float(conversion_ratio)
        if end_val is not None:
            valid_mask= times <= end_val
            times, data= times[valid_mask], data[valid_mask]


    #==== Option 2: Generate square waveform from start_times and end_times =====
    else:
        start_times= np.asarray(start_times)
        end_times= np.asarray(end_times)

        #=== Input validation ===
        if start_times.ndim != 1 or end_times.ndim != 1:
            raise ValueError("'start_times' and 'end_times' must be 1D.")
        if len(start_times) != len(end_times):
            raise ValueError("'start_times' and 'end_times' must have the same length.")
        if np.any(end_times < start_times):
            bad_idx = int(np.where(end_times < start_times)[0][0])
            raise ValueError(f"End time is earlier than start time at index {bad_idx}.")

        #=== Create time vector ===
        t0, t1 = np.min(start_times), np.max(end_times)
        times = np.arange(t0, t1 + resolution, resolution)

        #=== Create square waveform ===
        data = np.zeros_like(times)
        for start, end in zip(start_times, end_times):
            data[(times >= start) & (times <= end)] = 100


    #==== Build run-length segments =============================================
    segments = []
    changes = np.where(np.diff(data, prepend=data[0]) != 0)[0]

    # Ensure change boundaries include start and end of data
    boundaries = np.concatenate([changes, [len(data)]])

    # Set start and end indices for segments
    start_indices= boundaries[:-1]
    end_indices= boundaries[1:] - 1

    # Set state of segment to data value at start index
    states= data[start_indices]

    # Start and end times of segments
    seg_start_times= times[start_indices]
    seg_end_times= times[end_indices]

    # Durations
    durations= seg_end_times - seg_start_times

    # Build initial segment list
    for s, e, d, st in zip(seg_start_times, seg_end_times, durations, states):
        segments.append({'Start': s,
                         'End': e,
                         'Duration': d,
                         'State': st})


    #==== Filter segments by minimum separation =================================

    if min_separation is not None and min_separation > 0:
        filtered_segments = []
        i = 0

        while i < len(segments):
            current_segment = segments[i].copy()

            if current_segment['Duration'] < min_separation:
                # Merge with next segment if possible
                if i + 1 < len(segments):
                    next_segment = segments[i + 1]
                    if next_segment['Start'] - current_segment['End'] < min_separation:
                        current_segment['End'] = next_segment['End']
                        current_segment['Duration'] = current_segment['End'] - current_segment['Start']
                        current_segment['State'] = next_segment['State']
                        segments.pop(i + 1)
                    else:
                        i += 1
                else:
                    i += 1
            else:
                filtered_segments.append(current_segment)
                i += 1

        segments = filtered_segments


    #==== Build Final Cleaned Data and Pulse Dataframe ==========================
    new_data= np.zeros_like(data)
    for seg in segments:
        new_data[(times >= seg['Start']) & (times <= seg['End'])] = seg['State']

    data= new_data
    pulse_durations= pd.DataFrame(segments, columns= ['Start', 'End', 'Duration', 'State'])

    #==== Align to zero if specified ============================================
    time_offset= pulse_durations['Start'].iloc[0] if len(pulse_durations) > 0 else 0.0

    if align_to_zero is True:
        if len(pulse_durations) > 0:
            pulse_durations['Start']= pulse_durations['Start'] - time_offset
            pulse_durations['End']= pulse_durations['End'] - time_offset

    #==== Round time values if specified ========================================
    if decimal_places is not None:
        times= np.round(times, decimals= decimal_places)
        pulse_durations['Start']= np.round(pulse_durations['Start'], decimals= decimal_places)
        pulse_durations['End']= np.round(pulse_durations['End'], decimals= decimal_places)
        pulse_durations['Duration']= np.round(pulse_durations['Duration'], decimals= decimal_places)

    #==== Drop inactive segments if specified ===================================
    if df_active_only is True:
        pulse_durations= pulse_durations[pulse_durations['State'] != 0].reset_index(drop= True)

    return times, time_offset, data, pulse_durations


#===============================================================================




#===============================================================================
# 6| DigitalSyncPulse
#===============================================================================


class DigitalSyncPulse:
    '''
    Extracts digital sync pulses from a specified Harp Behavior device for
    neuropixel alignment.

    Reads DO1 rise (register 34) and fall (register 35) events from the
    specified sync device, pairs them into pulse start/end times, and builds
    a ``SyncPulseTimes`` DataFrame.

    Parameters
    ----------
    experiment_directory_path : str
        Path to the experiment directory.
    harp_device_yaml_path : str
        Path to the harp device YAML file.
    device_type : str
        Type prefix for device folders (e.g. 'Behavior').
    Sync_device : str or list
        Name(s) of the device used for synchronisation. If a list, the first
        element is used.
    device_list : list of str
        Full list of device names to load.
    device_IDs : dict
        Dictionary mapping device names to their IDs.
    device_registers_dict_str : list of str
        Register addresses for DO1 rise and fall. Default is ['34', '35'].
    min_duration : float
        Minimum duration for a valid pulse. Default is 0.0.
    align_to_zero : bool
        If True, normalises pulse times so the first pulse starts at 0.
    verbose : bool
        If True, prints verbose output.

    Attributes
    ----------
    SyncPulseTimes : pd.DataFrame
        DataFrame with Start, End, Duration columns for each detected pulse.
    SyncPulseTimes_unaligned : pd.DataFrame or None
        Original (un-normalised) pulse times if align_to_zero is True.
    initial_start_offset : float
        Start time of the first pulse in original timebase.
    '''
    def __init__(self,
                 experiment_directory_path= None,
                 harp_device_yaml_path= None,
                 device_type= 'Behavior',
                 Sync_device= 'Behavior3',
                 device_list= None,
                 device_IDs= None,
                 device_registers_dict_str= None,
                 min_duration= 0.0,
                 align_to_zero= True,
                 verbose= False):
        super(DigitalSyncPulse, self).__init__()

        ##===== User-Defined Parameters ========================================

        self.experimental_directory_path= experiment_directory_path
        self.harp_device_yaml_path= harp_device_yaml_path

        self.device_type= device_type
        self.Sync_device= Sync_device
        self.device_list= device_list
        self.device_IDs= device_IDs if device_IDs is not None else {}

        if device_registers_dict_str is None:
            device_registers_dict_str = ['34', '35']
        self.device_registers_dict_str= device_registers_dict_str

        # Build per-device register dict from the sync device name
        sync_name = self.Sync_device[0] if isinstance(self.Sync_device, list) else self.Sync_device
        self.device_registers_dict= {sync_name: device_registers_dict_str}

        self.align_to_zero= align_to_zero
        self.min_duration= min_duration
        self.verbose= verbose


        ##===== Setup ==========================================================

        # Lazy imports — use refactored harp_extender_core when available,
        # otherwise fall back to legacy Helper_functions.
        try:
            from Refactor.HarpExtender.harp_extender_core import (
                construct_device_reader as _construct_device_reader,
                collect_device_folders as _collect_device_folders,
                collect_device_dfs as _collect_device_dfs,
            )
        except ImportError:
            from Data_Imports.Helper_Functions.Helper_functions import (
                create_device_reader as _construct_device_reader,
                get_device_subfolders as _collect_device_folders,
                get_dict_of_devices_register_dfs_dicts as _collect_device_dfs,
            )

        self.harp_device_reader= _construct_device_reader(
            harp_device_yaml_path= self.harp_device_yaml_path
        )

        self.device_folders= _collect_device_folders(
            experiment_directory_path= self.experimental_directory_path,
            device_type= self.device_type
        )

        self.dict_of_devices_register_dfs_dicts= _collect_device_dfs(
            experiment_directory_path= self.experimental_directory_path,
            harp_device_yaml_path= self.harp_device_yaml_path,
            device_type= self.device_type,
            device_list= self.device_list,
            device_IDs= self.device_IDs,
            device_registers_dict= self.device_registers_dict,
        )


        ##===== Extract Event Times ============================================

        # Extract DO1 rise/start and fall/end times from specified registers
        start_series= self.dict_of_devices_register_dfs_dicts[sync_name][self.device_registers_dict_str[0]]['DO1']
        end_series= self.dict_of_devices_register_dfs_dicts[sync_name][self.device_registers_dict_str[1]]['DO1']

        def _extract_event_times(s):
            '''Convert input to a Series of event timestamps.'''
            if isinstance(s, pd.DataFrame):
                s = s.squeeze()
            if isinstance(s, pd.Series):
                if s.dtype == bool or np.issubdtype(s.dtype, np.integer) or np.issubdtype(s.dtype, np.floating):
                    nonzero = s.astype(bool)
                    unique_vals = pd.unique(s[nonzero])
                    if len(unique_vals) <= 2:
                        times = s.index[nonzero]
                    else:
                        times = s.dropna().values
                else:
                    times = s.dropna().values
            else:
                times = pd.Series(s).dropna().values
            return pd.Series(times)

        start_times = _extract_event_times(start_series)
        end_times = _extract_event_times(end_series)

        # Filter out pulses that do not meet minimum duration criteria
        n= int(min(len(start_times), len(end_times)))
        valid_mask = (end_times.iloc[:n].to_numpy() - start_times.iloc[:n].to_numpy()) >= min_duration
        start_times = start_times.iloc[:n][valid_mask]
        end_times = end_times.iloc[:n][valid_mask]
        n = len(start_times)

        self.SyncPulseTimes= pd.DataFrame(
            {'Start': start_times.to_numpy(),
             'End': end_times.to_numpy(),
             'Duration': end_times.to_numpy() - start_times.to_numpy()},
            index= pd.RangeIndex(1, n + 1, name='Pulse')
        )
        self.initial_start_offset= self.SyncPulseTimes['Start'].iloc[0]

        self.SyncPulseTimes_unaligned = None
        if self.align_to_zero:
            self.SyncPulseTimes_unaligned= self.SyncPulseTimes.copy()
            offset = self.SyncPulseTimes['Start'].iloc[0]
            self.SyncPulseTimes= pd.DataFrame(
                {'Start': self.SyncPulseTimes['Start'] - offset,
                 'End': self.SyncPulseTimes['End'] - offset,
                 'Duration': self.SyncPulseTimes['Duration']}
            )
            if verbose:
                print('Warning: If using align_to_zero=True, SyncPulseTimes are aligned to start at 0. '
                      'Use initial_start_offset to recover original timestamps.')


#===============================================================================




#===============================================================================
# 7| Npx_SyncPulse
#===============================================================================


class Npx_SyncPulse:
    '''
    Extracts Neuropixel sync pulses from a .npy file using
    ``get_square_waveform``.

    Parameters
    ----------
    npx_path : str or None
        Path to NPX sync channel .npy file.
    npx_data : array-like or None
        Pre-loaded NPX sync channel data. If provided, npx_path is ignored.
    npx_frequency : float or None
        Sampling frequency for NPX data. Conversion ratio is 1/npx_frequency.
        Default is 1 (no conversion).
    start_val : float or None
        Starting time offset. Default is None (0).
    end_val : float or None
        End time for trimming. Default is None (no trimming).
    decimal_places : int or None
        Decimal places for rounding. Default is None (no rounding).
    min_separation : float or None
        Minimum separation between pulses. Default is None (no minimum).
    align_to_zero : bool
        Whether to align the first pulse to time zero. Default is False.
    df_active_only : bool
        If True, only active segments are kept. Default is True.
    verbose : bool
        Whether to print verbose output. Default is False.

    Attributes
    ----------
    npx_times : np.ndarray
        Time values for the NPX sync channel waveform.
    time_offset : float
        Time offset of the first pulse start (before zero-alignment).
    npx_data : np.ndarray
        NPX sync channel waveform data (0 for low, 100 for high).
    npx_pulse_durations : pd.DataFrame
        DataFrame of pulse Start, End, Duration, State.
    modified_npx_pulse_durations : pd.DataFrame
        Same as npx_pulse_durations but with the first pulse removed
        (for alignment purposes).
    npx_pulse_durations_original : pd.DataFrame
        Pulse durations in original time base (before zero-alignment).
    syncpulse_data : dict
        Dictionary collecting all outputs for convenient access.
    '''
    def __init__(self,
                 npx_path= None,
                 npx_data= None,
                 npx_frequency= 1,
                 start_val= None,
                 end_val= None,
                 decimal_places= None,
                 min_separation= None,
                 align_to_zero= False,
                 df_active_only= True,
                 verbose= False):
        super(Npx_SyncPulse, self).__init__()

        self.npx_path= npx_path
        self.npx_data= npx_data
        self.npx_frequency= npx_frequency
        self.start_val= start_val
        self.end_val= end_val
        self.decimal_places= decimal_places
        self.min_separation= min_separation
        self.align_to_zero= align_to_zero
        self.df_active_only= df_active_only
        self.verbose= verbose

        #== Get NPX data if not provided
        if self.npx_data is None:
            if self.npx_path is None:
                raise ValueError("Either 'npx_path' or 'npx_data' must be provided.")
            else:
                self.npx_data= np.load(self.npx_path, allow_pickle= True)

        #== Determine conversion ratio
        self.conversion_ratio= 1 / self.npx_frequency if self.npx_frequency is not None else 1

        #== Use get_square_waveform to extract pulses
        self.npx_times, self.time_offset, self.npx_data, self.npx_pulse_durations= get_square_waveform(
            oneD_data= self.npx_data,
            start_times= None,
            end_times= None,
            resolution= None,
            start_val= self.start_val,
            end_val= self.end_val,
            conversion_ratio= self.conversion_ratio,
            min_separation= self.min_separation,
            decimal_places= self.decimal_places,
            align_to_zero= self.align_to_zero,
            df_active_only= self.df_active_only,
        )

        #== Verify detected peaks
        from scipy.signal import find_peaks as _find_peaks
        npx_peaks, _ = _find_peaks(self.npx_data, height=0)
        if self.verbose:
            print(f'Number of detected peaks: {len(npx_peaks)}')
        if len(npx_peaks) != len(self.npx_pulse_durations):
            print("Warning: Number of detected peaks does not match number of pulse durations. "
                  "Something may be wrong. Please confirm the data is correct.")

        if self.verbose:
            print(self.npx_pulse_durations)

        #== Create modified pulse durations without first pulse for alignment
        self.modified_npx_pulse_durations= self.npx_pulse_durations.iloc[1:].reset_index(drop= True)

        #== Store original pulse durations before zero alignment
        if self.align_to_zero:
            self.npx_pulse_durations_original= pd.DataFrame()
            self.npx_pulse_durations_original['Start']= self.npx_pulse_durations['Start'] + self.time_offset
            self.npx_pulse_durations_original['End']= self.npx_pulse_durations['End'] + self.time_offset
            self.npx_pulse_durations_original['Duration']= self.npx_pulse_durations['Duration']
        else:
            self.npx_pulse_durations_original= self.npx_pulse_durations.copy()

        if self.verbose:
            print("Original NPX pulse durations (before zero alignment):")
            print(self.npx_pulse_durations_original)

        #== Rename index to 'Pulse' for consistency with DigitalSyncPulse
        self.npx_pulse_durations.index.name = 'Pulse'
        self.modified_npx_pulse_durations.index.name = 'Pulse'
        self.npx_pulse_durations_original.index.name = 'Pulse'

        self.syncpulse_data= {
            'npx_times': self.npx_times,
            'time_offset': self.time_offset,
            'npx_data': self.npx_data,
            'npx_pulse_durations': self.npx_pulse_durations,
            'modified_npx_pulse_durations': self.modified_npx_pulse_durations,
            'npx_pulse_durations_original': self.npx_pulse_durations_original
        }


#===============================================================================




########################################################################################################################
# Sync Pulse Alignment & Conversion
########################################################################################################################




#===============================================================================
# 8| get_aligned_pulse_dfs
#===============================================================================


def get_aligned_pulse_dfs(sync_df= None,
                          npx_df= None,
                          verbose= False,
                          visualisation= False):
    '''
    Aligns Bonsai/Harp and NPX sync pulse DataFrames by matching the number of
    pulses and normalising t=0 to the start of the first pulse.

    Parameters
    ----------
    sync_df : pd.DataFrame
        Bonsai/Harp sync pulse times with Start, End, Duration columns.
    npx_df : pd.DataFrame
        NPX sync pulse times with Start, End, Duration columns.
    verbose : bool
        If True, prints verbose output.
    visualisation : bool
        If True, plots stacked pulses for visual verification.

    Returns
    -------
    aligned_sync : pd.DataFrame
        Bonsai/Harp sync times normalised to first pulse.
    aligned_npx : pd.DataFrame
        NPX sync times normalised to first pulse.
    '''
    # Keep matching number of pulses
    n_keep = min(len(sync_df), len(npx_df))
    aligned_sync = sync_df.iloc[:n_keep].copy().reset_index(drop=True)
    aligned_npx = npx_df.iloc[:n_keep].copy().reset_index(drop=True)

    # Normalise to first pulse
    s0 = float(aligned_sync['Start'].iloc[0])
    aligned_sync.loc[:, ['Start', 'End']] = aligned_sync.loc[:, ['Start', 'End']] - s0

    n0 = float(aligned_npx['Start'].iloc[0])
    aligned_npx.loc[:, ['Start', 'End']] = aligned_npx.loc[:, ['Start', 'End']] - n0

    # Brief checks
    if verbose:
        print(f"Using {n_keep} pulses. Sync start range: {aligned_sync['Start'].iloc[0]} -> {aligned_sync['End'].iloc[-1]}")
        print(f"Using {n_keep} pulses. Npx start range: {aligned_npx['Start'].iloc[0]} -> {aligned_npx['End'].iloc[-1]}")

    # Visual verification
    if visualisation:
        plot_stacked_pulses(sync_df= aligned_sync,
                            npx_df= aligned_npx,
                            pulses_per_row= None,
                            autofit_pulses_per_row= True)

    return aligned_sync, aligned_npx


#===============================================================================




#===============================================================================
# 9| get_npx_to_bonsai_time_conversion
#===============================================================================


def get_npx_to_bonsai_time_conversion(sync_df= None,
                                      npx_df= None,
                                      visualisation= False,
                                      error_visualisation_tools= None,
                                      method= 'Range',
                                      replace= False,
                                      verbose= False):
    '''
    Computes the conversion ratio from NPX samples to Bonsai/Harp sync time
    and optionally converts the NPX pulse DataFrame.

    Parameters
    ----------
    sync_df : pd.DataFrame
        Bonsai/Harp sync pulse times with Start, End, Duration columns.
    npx_df : pd.DataFrame
        NPX sync pulse times with Start, End, Duration columns.
    visualisation : bool
        If True, plots stacked pulses after conversion.
    error_visualisation_tools : any or None
        If truthy, plots error scatter / histograms.
    method : str
        Conversion method. Currently only 'Range' is implemented.
    replace : bool
        If True, modifies npx_df in place and returns only the ratio.
    verbose : bool
        If True, prints verbose output.

    Returns
    -------
    aligned_npx_df_converted : pd.DataFrame
        NPX pulse times converted to Bonsai time. Returned only if
        replace is False.
    npx_conversion_ratio : float
        The computed conversion ratio (NPX_time / Bonsai_time).
    '''
    import matplotlib.pyplot as plt

    #=== Compute conversion ratio ==============================================
    if method == 'Range':
        aligned_sync_df, aligned_npx_df = get_aligned_pulse_dfs(
            sync_df= sync_df,
            npx_df= npx_df,
            verbose= verbose,
            visualisation= False,
        )

        npx_conversion_ratio = (
            (aligned_npx_df['End'].iloc[-1] - aligned_npx_df['Start'].iloc[0]) /
            (aligned_sync_df['End'].iloc[-1] - aligned_sync_df['Start'].iloc[0])
        )

        if verbose:
            print(f"Computed NPX to Bonsai ratio (NPX_time/Bonsai_time): {npx_conversion_ratio}.\n"
                  f"Convert using 1 / {npx_conversion_ratio} to get Bonsai_time from NPX_time.\n")
            print(f'NPX time range: {aligned_npx_df["Start"].iloc[0]} to {aligned_npx_df["End"].iloc[-1]} units.\n'
                  f'Total duration: {aligned_npx_df["End"].iloc[-1] - aligned_npx_df["Start"].iloc[0]} units for {len(aligned_npx_df)} pulses.\n')
            print(f'Bonsai time range: {aligned_sync_df["Start"].iloc[0]} to {aligned_sync_df["End"].iloc[-1]} seconds.\n'
                  f'Total duration: {aligned_sync_df["End"].iloc[-1] - aligned_sync_df["Start"].iloc[0]} seconds for {len(aligned_sync_df)} pulses.\n')
    else:
        raise NotImplementedError("Currently only 'Range' method is implemented for conversion ratio computation.")


    #=== Convert NPX times to Bonsai times ====================================
    aligned_npx_df_converted= aligned_npx_df.copy()
    aligned_npx_df_converted['Start']= aligned_npx_df_converted['Start'] * (1 / npx_conversion_ratio)
    aligned_npx_df_converted['End']= aligned_npx_df_converted['End'] * (1 / npx_conversion_ratio)
    aligned_npx_df_converted['Duration']= aligned_npx_df_converted['Duration'] * (1 / npx_conversion_ratio)

    if verbose:
        print("Converted NPX pulse times to Bonsai times:")
        print(aligned_npx_df_converted)

    if replace:
        npx_df['Start']= npx_df['Start'] * (1 / npx_conversion_ratio)
        npx_df['End']= npx_df['End'] * (1 / npx_conversion_ratio)
        npx_df['Duration']= npx_df['Duration'] * (1 / npx_conversion_ratio)
        if verbose:
            print("Replaced original NPX dataframe times with converted Bonsai times.")


    #=== Error Calculation =====================================================

    start_time_errors= aligned_npx_df_converted['Start'] - aligned_sync_df['Start']
    start_time_errors_ms= start_time_errors * 1000

    end_time_errors= aligned_npx_df_converted['End'] - aligned_sync_df['End']
    end_time_errors_ms= end_time_errors * 1000

    duration_errors= aligned_npx_df_converted['Duration'] - aligned_sync_df['Duration']
    duration_errors_ms= duration_errors * 1000

    error_df= pd.DataFrame({
        'Start_Time_Error_ms': start_time_errors_ms,
        'End_Time_Error_ms': end_time_errors_ms,
        'Duration_Error_ms': duration_errors_ms,
    })

    if verbose:
        print("Error Analysis (in milliseconds):")
        print(error_df)
        print(f"Start Time Errors (ms):\n  Mean: {np.mean(start_time_errors_ms):.6f}, "
              f"Std: {np.std(start_time_errors_ms):.6f}, "
              f"Max: {np.max(start_time_errors_ms):.6f}, "
              f"Min: {np.min(start_time_errors_ms):.6f}\n")
        print(f"End Time Errors (ms):\n  Mean: {np.mean(end_time_errors_ms):.6f}, "
              f"Std: {np.std(end_time_errors_ms):.6f}, "
              f"Max: {np.max(end_time_errors_ms):.6f}, "
              f"Min: {np.min(end_time_errors_ms):.6f}\n")
        print(f"Duration Errors (ms):\n  Mean: {np.mean(duration_errors_ms):.6f}, "
              f"Std: {np.std(duration_errors_ms):.6f}, "
              f"Max: {np.max(duration_errors_ms):.6f}, "
              f"Min: {np.min(duration_errors_ms):.6f}\n")


    #=== Visual verification ===================================================
    if visualisation:
        plot_stacked_pulses(sync_df= aligned_sync_df,
                            npx_df= aligned_npx_df_converted,
                            pulses_per_row= None,
                            autofit_pulses_per_row= True)

    if error_visualisation_tools is not None and error_visualisation_tools is not False:
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 9), sharex='col')

        idx = aligned_npx_df_converted.index
        errors = [
            ("Start", start_time_errors_ms),
            ("End", end_time_errors_ms),
            ("Duration", duration_errors_ms),
        ]

        for i, (label, data) in enumerate(errors):
            ax_scatter = axs[i, 0]
            ax_hist = axs[i, 1]

            ax_scatter.scatter(idx, data, s=10, alpha=0.7,
                               label=f"{label} Time Error (ms)", edgecolors='none')
            ax_scatter.set_ylabel("Error (ms)")
            if np.max(np.abs(data)) < 1:
                ax_scatter.set_ylim(-1, 1)
            ax_scatter.legend(loc="upper right", fontsize="small")
            ax_scatter.grid(True, alpha=0.2)

            ax_hist.hist(data, bins=30, alpha=0.7,
                         label=f"{label} Time Error (ms)")
            ax_hist.set_ylabel("Count")
            ax_hist.legend(loc="upper right", fontsize="small")
            ax_hist.grid(True, alpha=0.2)

        axs[-1, 0].set_xlabel("Pulse Index")
        axs[-1, 1].set_xlabel("Error (ms)")
        axs[0, 0].set_title("Error vs pulse index")
        axs[0, 1].set_title("Error distribution")

        plt.tight_layout()
        plt.show()


    #=== Return results ========================================================
    if replace:
        return npx_conversion_ratio
    else:
        return aligned_npx_df_converted, npx_conversion_ratio


#===============================================================================




#===============================================================================
# 10| convert_npx_to_bonsai_time
#===============================================================================


def convert_npx_to_bonsai_time(npx_data= None,
                               column_names= None,
                               npx_sync_pulse= None,
                               sync_pulses= None,
                               npx_pulse_start= None,
                               sync_pulse_start= None,
                               npx_to_bonsai_ratio= None,
                               verbose= False,
                               in_place= False):
    '''
    Convert NPX times to equivalent Bonsai times.

    Option 1: Pass ``npx_sync_pulse`` and ``sync_pulses`` instances;
    the function will compute the conversion parameters automatically.

    Option 2: Provide ``npx_pulse_start``, ``syncpulse_start``, and
    ``npx_to_bonsai_ratio`` directly.

    Parameters
    ----------
    npx_data : pd.DataFrame, pd.Series, np.ndarray, or list
        NPX times to convert. Can be a DataFrame (converts specified columns),
        a Series, or a 1D array.
    column_names : list of str or None
        If npx_data is a DataFrame, specify which columns to convert.
        Default is None (all columns).
    npx_sync_pulse : Npx_SyncPulse or None
        Instance of Npx_SyncPulse class.
    sync_pulses : DigitalSyncPulse or None
        Instance of DigitalSyncPulse class.
    npx_pulse_start : float or None
        Start time of NPX first sync pulse.
    syncpulse_start : float or None
        Start time of Bonsai first sync pulse.
    npx_to_bonsai_ratio : float or None
        Conversion ratio (NPX_time / Bonsai_time).
    verbose : bool
        If True, prints verbose output.
    in_place : bool
        If True and npx_data is a DataFrame, modifies it in place.

    Returns
    -------
    converted_npx_data : same type as npx_data
        NPX times converted to Bonsai times. Not returned if in_place is True.
    '''

    #=== Determine conversion parameters =======================================
    if npx_sync_pulse is not None and sync_pulses is not None:
        npx_pulse_start = npx_sync_pulse.npx_pulse_durations['Start'].iloc[0]
        syncpulse_start = sync_pulses.SyncPulseTimes['Start'].iloc[0]
        _, npx_to_bonsai_ratio = get_npx_to_bonsai_time_conversion(
            sync_df= sync_pulses.SyncPulseTimes,
            npx_df= npx_sync_pulse.npx_pulse_durations,
            visualisation= False,
            error_visualisation_tools= None,
            method= 'Range',
            replace= False,
            verbose= verbose
        )
        if verbose:
            print(f"Using Npx_SyncPulse and SyncPulses classes for conversion.\n"
                  f"NPX sync pulse start: {npx_pulse_start}\n"
                  f"Bonsai sync pulse start: {syncpulse_start}\n"
                  f"Conversion ratio: {npx_to_bonsai_ratio}\n")

    #=== Conversion loop =======================================================

    if isinstance(npx_data, pd.DataFrame):
        converted_npx_data = npx_data.copy()
        columns_to_convert = column_names if column_names is not None else npx_data.columns.tolist()

        for col in columns_to_convert:
            if verbose:
                print(f"Converting column: {col}")

            if col == 'Duration':
                converted_npx_data[col] = npx_data[col] * (1 / npx_to_bonsai_ratio)
                continue
            if col == 'State':
                if verbose:
                    print(f"Skipping conversion for 'State' column.")
                continue

            converted_npx_data[col] = (
                (npx_data[col] - npx_pulse_start) * (1 / npx_to_bonsai_ratio) + syncpulse_start
            )

        if in_place:
            for col in columns_to_convert:
                npx_data[col] = converted_npx_data[col]
            if verbose:
                print("Converted NPX times updated in place in the original DataFrame.")
            return
        else:
            return converted_npx_data

    def _conversion_transformation(data):
        '''Apply conversion transformation to 1D array-like data.'''
        return (data - npx_pulse_start) * (1 / npx_to_bonsai_ratio) + syncpulse_start

    #=== Single column / array input ===========================================
    if isinstance(npx_data, pd.Series):
        return _conversion_transformation(npx_data)
    elif isinstance(npx_data, np.ndarray) or isinstance(npx_data, list):
        npx_data_array = np.asarray(npx_data)
        return _conversion_transformation(npx_data_array)


#===============================================================================




########################################################################################################################
# Visualisation
########################################################################################################################




#===============================================================================
# 11| generate_waveform_from_start_end_times
#===============================================================================


def generate_waveform_from_start_end_times(start_times= None,
                                           end_times= None,
                                           oneD_data= None,
                                           oneD_times= None,
                                           resolution= 1,
                                           show_plot= True,
                                           peak_trough_detection= True,
                                           verbose= False,
                                           Return_start_end_times= False):
    '''
    Generates a visualisation of a square waveform from start and end times or
    from 1D timeseries data.

    Parameters
    ----------
    start_times : array-like or None
        Start times of square waves.
    end_times : array-like or None
        End times of square waves.
    oneD_data : array-like or None
        1D timeseries data containing square waves.
    oneD_times : array-like or None
        Time axis for oneD_data. If None, uses np.arange(len(oneD_data)).
    resolution : float
        Time resolution when generating from start/end times.
    show_plot : bool
        Whether to display the plot. Default is True.
    peak_trough_detection : bool
        Whether to perform peak and trough detection. Default is True.
    verbose : bool
        Whether to print verbose output.
    Return_start_end_times : bool
        Whether to return detected pulse start/end times. Default is False.

    Returns
    -------
    waveform : np.ndarray
        (N x 2) array with time and data values.
    figure : matplotlib.figure.Figure
        Returned only if show_plot is True.
    pulse_durations : pd.DataFrame
        Returned only if Return_start_end_times is True.
    '''
    import matplotlib.pyplot as plt

    if oneD_data is None and (start_times is None or end_times is None):
        raise ValueError("Either 'oneD_data' or both 'start_times' and 'end_times' must be provided.")

    if oneD_data is not None:
        if oneD_times is None:
            oneD_times = np.arange(len(oneD_data))
            if verbose:
                print("oneD_times not provided. Generating default time vector based on length of oneD_data.")
        time_vector = np.asarray(oneD_times)
        data_vector = np.asarray(oneD_data)
        data_vector = np.where(data_vector > 0, 100, 0)
        waveform = np.column_stack((time_vector, data_vector))

    elif start_times is not None and end_times is not None and oneD_data is None:
        time_vector = np.arange(np.min(start_times), np.max(end_times) + resolution, resolution)
        data_vector = np.zeros_like(time_vector)

        for start, end in zip(start_times, end_times):
            data_vector[(time_vector >= start) & (time_vector <= end)] = 100

        waveform = np.column_stack((time_vector, data_vector))

    # Peak and trough detection
    if peak_trough_detection is True:
        peaks= np.where((data_vector[1:] == 100) & (data_vector[:-1] == 0))[0] + 1
        troughs= np.where((data_vector[1:] < 100) & (data_vector[:-1] == 100))[0]
        if verbose:
            print(f'Number of detected peaks: {len(peaks)}')
            print(f'Number of detected troughs: {len(troughs)}')

    # Plot waveform
    fig = None
    if show_plot is True:
        fig, ax = plt.subplots(figsize=(72, 4))
        ax.plot(waveform[:, 0], waveform[:, 1], label='Square Waveform', color='k', alpha=0.9)
        if peak_trough_detection is True:
            ax.scatter(waveform[:, 0][peaks], waveform[:, 1][peaks], marker='x', label='Start', color='green')
            ax.scatter(waveform[:, 0][troughs], waveform[:, 1][troughs], marker='x', label='End', color='red')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Square Waveform from Start and End Times')
        ax.legend()
        ax.set_facecolor('whitesmoke')
        plt.grid()
        plt.show()

    if Return_start_end_times is True:
        pulse_durations= pd.DataFrame({
            'Start': waveform[:, 0][peaks],
            'End': waveform[:, 0][troughs],
        })
        pulse_durations['Duration']= pulse_durations['End'] - pulse_durations['Start']
        return waveform, fig, pulse_durations
    else:
        return waveform


#===============================================================================




#===============================================================================
# 12| plot_stacked_pulses
#===============================================================================


def plot_stacked_pulses(sync_df= None,
                        npx_df= None,
                        pulses_per_row= None,
                        autofit_pulses_per_row= True):
    '''
    Plots stacked Bonsai/Harp and NPX sync pulses for visual comparison.

    Parameters
    ----------
    sync_df : pd.DataFrame
        Bonsai/Harp sync pulse times with 'Start' and 'End' or 'Duration'
        columns.
    npx_df : pd.DataFrame
        NPX sync pulse times with 'Start' and 'End' or 'Duration' columns.
    pulses_per_row : int or None
        Number of pulses to display per row. If None, determined automatically.
    autofit_pulses_per_row : bool
        If True and pulses_per_row is None, determines pulses_per_row from
        the largest factor of the total pulse count. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object, or None if no pulses.
    axes : list of matplotlib.axes.Axes or None
        The axes objects, or None if no pulses.
    '''
    import matplotlib.pyplot as plt

    # Prepare (start, duration) arrays
    def prep(df):
        df = df.copy()
        if 'State' in df.columns:
            df = df[df['State'] == 100]
        cols = {c.lower(): c for c in df.columns}
        s = cols.get('start') or cols.get('start_time')
        e = cols.get('end') or cols.get('end_time')
        d = cols.get('duration')
        if s is None:
            raise ValueError("Missing Start/Start_Time column.")
        starts = df[s].astype(float).to_numpy()
        if d is not None:
            durs = df[d].astype(float).to_numpy()
        elif e is not None:
            durs = (df[e] - df[s]).astype(float).to_numpy()
        else:
            raise ValueError("Need either Duration or End column.")
        durs = np.maximum(durs, 0.0)
        return starts, durs

    sp_starts, sp_durs = prep(sync_df)
    np_starts, np_durs = prep(npx_df)

    n = int(max(len(sp_starts), len(np_starts)))
    if n == 0:
        print("No pulses to plot.")
        return None, None

    # Determine pulses_per_row
    if pulses_per_row is None and autofit_pulses_per_row:
        from sympy import primefactors
        pulses_to_use = min(len(np_starts), len(sp_starts))
        factors = primefactors(pulses_to_use)
        if len(factors) > 0:
            pulses_per_row = pulses_to_use // min(factors)
        else:
            pulses_per_row = min(len(sp_starts), len(np_starts))
    elif pulses_per_row is None:
        pulses_per_row = min(len(sp_starts), len(np_starts))

    n_rows = int(np.ceil(n / pulses_per_row))

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.2 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    for r in range(n_rows):
        ax = axes[r]
        i0 = r * pulses_per_row
        i1 = min(i0 + pulses_per_row, n)

        # Sync pulses (seconds) on primary axis
        seg_sync = list(zip(sp_starts[i0:i1], sp_durs[i0:i1]))
        ax.broken_barh(seg_sync, (0.2, 0.6), facecolors="#0463de")
        ax.set_xlabel('Sync time (s)')

        # Neuropixel pulses (samples) on twin x-axis
        ax2 = ax.twiny()
        seg_npx = list(zip(np_starts[i0:i1], np_durs[i0:i1]))
        ax2.broken_barh(seg_npx, (1.2, 0.6), facecolors='#de0404')
        ax2.set_xlabel('Neuropixel (samples)')

        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Sync', 'Neuropixel'])
        ax.set_ylim(0, 2)
        ax.set_title(f'Pulses {i0 + 1}\u2013{i1}')
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor("whitesmoke")
        ax2.set_facecolor("whitesmoke")

    return fig, axes


#===============================================================================
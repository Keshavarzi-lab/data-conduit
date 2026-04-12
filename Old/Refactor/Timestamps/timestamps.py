'''
timetamps.py
--------------------------------

Description: Module for handling timestamp-related functionalities in the context of device data collection and processing.

Contents:
--------------------------------
- collect_timestamps: Function to collect and process timestamps from device dataframes.

- collect_timestamps_dict: Function to collect and process timestamps from a dictionary of device dataframes.

- collect_timestamps_nested_dict: Function to collect and process timestamps from a nested dictionary of device dataframes.


'''


################################################################################
# Imports
################################################################################

import os
import re
from pathlib import Path
import harp
import pandas as pd
import numpy as np
import xarray as xr
import yaml

################################################################################




################################################################################
# Collect Timestamps
################################################################################


def collect_timestamps(df: pd.DataFrame | xr.DataArray | None = None,
                          timestamp_name: str | None = None,
                          time_location: str | None = None,
                          verbose: bool = True,
                          ) -> pd.Series | None:
    """
    Collects and processes timestamps from a given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame | xr.DataArray | None
        The DataFrame or DataArray containing the timestamps.
    timestamp_name : str | None
        The name of the column containing the timestamps. If None, defaults to 'Time'. Note: case-insensitive.
    time_location : str | None
        The location of the timestamps, either 'columns' or 'index'. If None, will attempt index first, then columns.
    verbose : bool
        If True, prints warnings and information during processing.

    Returns:
    --------
    pd.Series | None
        A Series of processed timestamps, or None if the DataFrame is None.
    """
    # 1. Check df 
    if df is None:
        return None

    # 2. Type Check
    if isinstance(df, xr.DataArray):
        df = df.to_dataframe().reset_index()
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame or xarray DataArray.")
    
    # 3. Set default timestamp column
    if not isinstance(timestamp_name, str):
        timestamp_name = 'Time' 
    
    def _find_column(columns, name):
        """
        Return the actual column name matching `name` (case-insensitive), or None.
        """
        for col in columns:
            if col.lower() == name.lower():
                return col
        return None


    # 4. Extract data (Check index, then columns)
    if time_location == 'index':
        # if df.index.name.lower() != timestamp_name.lower():
        #     if verbose: print(f"Timestamp name '{timestamp_name}' does not match DataFrame index name '{df.index.name}'.")
        #     return df.index
        
        index_name = df.index.name
        if index_name is not None and index_name.lower() == timestamp_name.lower():
            return df.index
        else:
            if verbose:
                print(f"Warning: Timestamp name '{timestamp_name}' does not match "
                      f"DataFrame index name '{index_name}'.")
            raise ValueError(
                f"Timestamp name '{timestamp_name}' not found in DataFrame index "
                f"(index name is '{index_name}')."
            )
    
    if time_location == 'columns':
        # if timestamp_name.lower() not in [col.lower() for col in df.columns]:   
        #     raise ValueError(f"Timestamp name '{timestamp_name}' not found in DataFrame columns.")

        # return df[timestamp_name]
        actual_col = _find_column(df.columns, timestamp_name)
        if actual_col is None:
            raise ValueError(f"Timestamp name '{timestamp_name}' not found in DataFrame columns.")
        return df[actual_col]
    
    if time_location is None:
        # Check index first
        if df.index.name is not None and df.index.name.lower() == timestamp_name.lower():
            return df.index
        # Then check columns
        # elif timestamp_name.lower() in [col.lower() for col in df.columns]:
        #     return df[timestamp_name]
        actual_col = _find_column(df.columns, timestamp_name)
        if actual_col is not None:
            return df[actual_col]
        else:
            raise ValueError(f"Timestamp name '{timestamp_name}' not found in DataFrame index or columns.")
    raise ValueError(f"Invalid time_location '{time_location}'. Must be 'index', 'columns', or None.")



################################################################################





################################################################################
# Collect Timestamps Dictionary
################################################################################

def collect_timestamps_dict(dfs: dict[str, pd.DataFrame | xr.DataArray],
                            timestamp_name: str | None = None,
                            time_location: str | None = None,
                            verbose: bool = True,
                            return_type: str = 'list',  # 'dict' or 'list'
                            ) -> dict[str, pd.Series] | list[pd.Series]:
    """
    Extract timestamps from each DataFrame in a dictionary.

    Parameters:
        dfs (dict): A dictionary where keys are identifiers and values are pandas DataFrames.
        timestamp_name (str, optional): The name of the column containing the timestamps. If None, defaults to 'Time'. Note: case-insensitive.
        verbose (bool, optional): If True, prints warnings and information during processing. Default is True.
        return_type (str, optional): If 'list', returns a sorted list of all timestamps. If 'dict', returns a dictionary of timestamps per DataFrame. Default is 'list'. If both are needed, call the function twice.

    Returns:
        list of all timestamps from each DataFrame in the dictionary, sorted in ascending order.
        dict where keys are the same as df_dict and values are the timestamps from each DataFrame.
    """ 

    if not isinstance(dfs, dict):
        raise TypeError("Input must be a dictionary of DataFrames.")
    
    return_type = return_type.lower()

    if return_type not in ['list', 'dict', 'both']:
        raise TypeError("return_type must be 'list', 'dict', or 'both'.")

    # if timestamp_name is None:
    #     timestamp_name = 'Time'
    #     if verbose: print(f'Warning! Timestamp checking is broken')
    
    # if return_type == 'list':
    #     all_timestamps= []

    #     for key, df in dfs.items():

    #         if not isinstance(df, (pd.DataFrame, xr.DataArray)):
    #             raise TypeError(f"Value for key '{key}' is not a pandas DataFrame or xarray DataArray.")
    #         if isinstance(df, xr.DataArray):
    #             df = df.to_dataframe().reset_index()
            
    #         # if df.index.name.lower() != timestamp_name.lower():
    #         #     if verbose: print(f"Timestamp name '{timestamp_name}' does not match DataFrame index name '{df.index.name}' for key '{key}'.")
            
    #         all_timestamps.extend(df.index)

    #     return sorted(set(all_timestamps))
    
    # elif return_type == 'dict':
    #     timestamps_dict = {}
        
    #     for key, df in dfs.items():
            
    #         if not isinstance(df, (pd.DataFrame, xr.DataArray)):
    #             raise TypeError(f"Value for key '{key}' is not a pandas DataFrame or xarray DataArray.")
    #         if isinstance(df, xr.DataArray):
    #             df = df.to_dataframe().reset_index()
            
    #         # if df.index.name.lower() != timestamp_name.lower():
    #         #     if verbose: print(f"Timestamp name '{timestamp_name}' does not match DataFrame index name '{df.index.name}' for key '{key}'.")
            
    #         timestamps_dict[key] = df.index

    #     return timestamps_dict

    # elif return_type == 'both':
    #     timestamps_dict = {}
    #     all_timestamps= []

    #     for key, df in dfs.items():
            
    #         if not isinstance(df, (pd.DataFrame, xr.DataArray)):
    #             raise TypeError(f"Value for key '{key}' is not a pandas DataFrame or xarray DataArray.")
    #         if isinstance(df, xr.DataArray):
    #             df = df.to_dataframe().reset_index()
            
    #         # if df.index.name.lower() != timestamp_name.lower():
    #         #     if verbose: print(f"Timestamp name '{timestamp_name}' does not match DataFrame index name '{df.index.name}' for key '{key}'.")
            
    #         timestamps_dict[key] = df.index
    #         all_timestamps.extend(df.index)

    #     return timestamps_dict, sorted(set(all_timestamps))
    timestamps_dict = {}
    for key, df in dfs.items():
        ts = collect_timestamps(
            df=df,
            timestamp_name=timestamp_name,
            time_location=time_location if time_location is not None else None,
            verbose=verbose,
        )
        if ts is not None:
            timestamps_dict[key] = ts

    # Return based on requested type
    if return_type == 'list':
        all_timestamps = []
        for ts in timestamps_dict.values():
            all_timestamps.extend(ts)
        return sorted(set(all_timestamps))

    elif return_type == 'dict':
        return timestamps_dict

    elif return_type == 'both':
        all_timestamps = []
        for ts in timestamps_dict.values():
            all_timestamps.extend(ts)
        return timestamps_dict, sorted(set(all_timestamps))


################################################################################




################################################################################
# Collect Timestamps Nested Dictionary
################################################################################

def collect_timestamps_nested_dict(dfs_dict: dict[str, dict[str, pd.DataFrame | xr.DataArray]],
                                   data_key: str | None = None,
                                   devices: list[str] | None = None,
                                #    peripheral_registerIDs: list[str] | None = None,
                                   peripheral_registerIDs: dict[str, dict[str, str]] | None = None,
                                   return_type: str = 'list', # 'dict' or 'list' or 'both'
                                   verbose: bool = True,
                                   ) -> dict[str, pd.Series] | list[pd.Series] | tuple[dict[str, pd.Series], list[pd.Series]]:
    '''
    Collect timestamps across all devices/registers relevant to the given type_key from the nested dfs_dict using collect_timestamps_dict.

    Parameters:
        dfs_dict (dict): dfs_dict {device: {register_address: DataFrame(Time index, localID columns)}}
        data_key (str): optional peripheral type to use (e.g., the global type_key) if using peripheral_registerIDs
        devices (list or None): optional subset of devices to include; defaults to all in dfs_dict
        peripheral_registerIDs (dict or None): optional subset of peripheral register IDs to include; defaults to all for each device
        return_type (str): 'list', 'dict', or 'both' (passed through to collect_timestamps_dict)
        verbose (bool): If True, prints warnings and information during processing.
    
    Returns:
        list | dict | (list, dict): timestamps per return_type
    '''
    # Guard clause
    if not isinstance(dfs_dict, dict) or not dfs_dict:
        print("Warning: dfs_dict is not a valid non-empty dictionary.")
        return [] if return_type == 'list' else ({} if return_type == 'dict' else ([], {}))
    

    #=== 1. Obtain Devices
    devices= devices if devices is not None else list(dfs_dict.keys())

    #=== 2. Filter registers based on devices and peripheral_registerIDs

    # if peripheral_registerIDs is not None and data_key is not None:
    #     reg_map= peripheral_registerIDs[data_key]
    #     needed_registers= sorted(set(reg_map.values()))

    # else:
    #      raise ValueError("Either peripheral_registerIDs and data_key must be provided together. Currently does not support collecting all registers without filtering.")
    # df_dict= {}  # Keyed by "{device}:{register}"

    # for device in devices:
    #     if device not in device_dfs_dict:
    #         print(f"Warning: Device {device} not found in device_dfs_dict; skipping.")
    #         continue

    #     dev_regs= device_dfs_dict[device]
    #     for reg in needed_registers:
    #         if reg not in dev_regs:
    #             print(f"Warning: Device {device} missing register {reg}; skipping.")
    #             continue

    #         # Restrict columns to localIDs relevant to this data_key and register
    #         localIDs_for_reg= [lid for lid, r in reg_map.items() if r == reg and lid in dev_regs[reg].columns]
    #         if not localIDs_for_reg:
    #             print(f"Warning: Device {device} register {reg} has no expected localIDs present; skipping.")
    #             continue

    #         df_subset= dev_regs[reg][localIDs_for_reg]
    #         df_dict[f"{device}:{reg}"]= df_subset
    df_dict = {}  # Keyed by "{device}:{register}"

    if peripheral_registerIDs is not None and data_key is not None:
        reg_map = peripheral_registerIDs[data_key]
        needed_registers = sorted(set(reg_map.values()))

        for device in devices:
            if device not in dfs_dict:
                if verbose:
                    print(f"Warning: Device {device} not found in dfs_dict; skipping.")
                continue

            dev_regs = dfs_dict[device]
            for reg in needed_registers:
                if reg not in dev_regs:
                    if verbose:
                        print(f"Warning: Device {device} missing register {reg}; skipping.")
                    continue

                localIDs_for_reg = [lid for lid, r in reg_map.items() if r == reg and lid in dev_regs[reg].columns]
                if not localIDs_for_reg:
                    if verbose:
                        print(f"Warning: Device {device} register {reg} has no expected localIDs present; skipping.")
                    continue

                df_dict[f"{device}:{reg}"] = dev_regs[reg][localIDs_for_reg]

    elif peripheral_registerIDs is None and data_key is None:
        for device in devices:
            if device not in dfs_dict:
                if verbose:
                    print(f"Warning: Device {device} not found in dfs_dict; skipping.")
                continue

            dev_regs = dfs_dict[device]
            for reg, df in dev_regs.items():
                df_dict[f"{device}:{reg}"] = df

    else:
        raise ValueError("peripheral_registerIDs and data_key must be provided together, or both left as None.")
    
    if not df_dict:
        print("Warning: No DataFrames collected for the specified data_key.")
        return [] if return_type == 'list' else ({} if return_type == 'dict' else ([], {}))
    if return_type == 'list':
        return collect_timestamps_dict(dfs=df_dict, return_type='list')
    elif return_type == 'dict':
        return collect_timestamps_dict(dfs=df_dict, return_type='dict')
    elif return_type == 'both':
        return collect_timestamps_dict(dfs=df_dict, return_type='both')
    else:
        raise ValueError("return_type must be one of 'list', 'dict', or 'both'.")

################################################################################




################################################################################################################################################################
# Wrappers for renamed, moved, or deprecated functions
################################################################################################################################################################


################################################################################
def get_df_timestamps(df: pd.DataFrame | xr.DataArray | None = None,
                      timestamp_name: str | None = None,
                      time_location: str | None = None,
                      verbose: bool = True,
                      ) -> pd.Series | None:
    """
    Wrapper for collect_df_timestamps function.

    Parameters:
    -----------
    df : pd.DataFrame | xr.DataArray | None
        The DataFrame or DataArray containing the timestamps.
    timestamp_name : str | None
        The name of the column containing the timestamps. If None, defaults to 'Time'. Note: case-insensitive.
    time_location : str | None
        The location of the timestamps, either 'columns' or 'index'. If None, will attempt index first, then columns.
    verbose : bool
        If True, prints warnings and information during processing.

    Returns:
    --------
    pd.Series | None
        A Series of processed timestamps, or None if the DataFrame is None.
    """
    print("Warning: 'get_df_timestamps' is deprecated. Please use 'collect_timestamps' instead.")
    return collect_timestamps(df=df,
                              timestamp_name=timestamp_name,
                              time_location=time_location,
                              verbose=verbose
                              )
##--------------------------------------------------------------------------------   Original
# def get_df_timestamps(df):
#     """
#     Extract timestamps from a DataFrame's index.

#     Parameters:
#         df (pd.DataFrame): The DataFrame from which to extract timestamps.

#     Returns:
#         pd.DatetimeIndex: The timestamps extracted from the DataFrame's index.
#     """
#     if not isinstance(df, pd.DataFrame):
#         raise ValueError("Input must be a pandas DataFrame.")
    
#     if df.index.name != 'Time':
#         raise ValueError("DataFrame index must be named 'Time'.")

#     return df.index
################################################################################





################################################################################
def get_df_dict_timestamps(df_dict: dict[str, pd.DataFrame | xr.DataArray],
                           timestamp_name: str | None = None,
                           verbose: bool = True,
                           return_type: str = 'list',  # 'dict' or 'list' or 'both'
                           ) -> dict[str, pd.Series] | list[pd.Series] | tuple[dict[str, pd.Series], list[pd.Series]]:
    """
    Wrapper for collect_timestamps_dict function.

    Parameters:
        df_dict (dict): A dictionary where keys are identifiers and values are pandas DataFrames.
        timestamp_name (str, optional): The name of the column containing the timestamps. If None, defaults to 'Time'. Note: case-insensitive.
        verbose (bool, optional): If True, prints warnings and information during processing. Default is True.
        return_type (str, optional): If 'list', returns a sorted list of all timestamps. If 'dict', returns a dictionary of timestamps per DataFrame. Default is 'list'. If both are needed, call the function twice.

    Returns:
        list of all timestamps from each DataFrame in the dictionary, sorted in ascending order.
        dict where keys are the same as df_dict and values are the timestamps from each DataFrame.
    """ 
    print("Warning: 'get_df_dict_timestamps' is deprecated. Please use 'collect_timestamps_dict' instead.")
    return collect_timestamps_dict(dfs=df_dict,
                                   timestamp_name=timestamp_name,
                                   verbose=verbose,
                                   return_type=return_type
                                   )

##--------------------------------------------------------------------------------   Original

# def get_df_dict_timestamps(df_dict, return_type='list'):
#     """
#     Extract timestamps from each DataFrame in a dictionary.

#     Parameters:
#         df_dict (dict): A dictionary where keys are identifiers and values are pandas DataFrames.
#         return_type (str, optional): If 'list', returns a sorted list of all timestamps. If 'dict', returns a dictionary of timestamps per DataFrame. Default is 'list'. If both are needed, call the function twice.

#     Returns:
#         list of all timestamps from each DataFrame in the dictionary, sorted in ascending order.
#         dict where keys are the same as df_dict and values are the timestamps from each DataFrame.
#     """
#     if not isinstance(df_dict, dict):
#         raise ValueError("Input must be a dictionary.")
    
#     if return_type is None:
#         raise ValueError("return_type must be specified as 'list' or 'dict'.")
    
#     if return_type == 'list':
#         all_timestamps = []
#         for key, df in df_dict.items():
#             if not isinstance(df, pd.DataFrame):
#                 raise ValueError(f"Value for key '{key}' is not a pandas DataFrame.")
#             if df.index.name != 'Time':
#                 raise ValueError(f"DataFrame for key '{key}' must have index named 'Time'.")
            
#             all_timestamps.extend(df.index)

#         return sorted(set(all_timestamps))
    
#     elif return_type == 'dict':
#         timestamps_dict = {}
#         for key, df in df_dict.items():
#             if not isinstance(df, pd.DataFrame):
#                 raise ValueError(f"Value for key '{key}' is not a pandas DataFrame.")
#             if df.index.name != 'Time':
#                 raise ValueError(f"DataFrame for key '{key}' must have index named 'Time'.")
            
#             timestamps_dict[key] = df.index

#         return timestamps_dict
    
#     elif return_type == 'both':
#         all_timestamps = []
#         timestamps_dict = {}
#         for key, df in df_dict.items():
#             if not isinstance(df, pd.DataFrame):
#                 raise ValueError(f"Value for key '{key}' is not a pandas DataFrame.")
#             if df.index.name != 'Time':
#                 raise ValueError(f"DataFrame for key '{key}' must have index named 'Time'.")

#             all_timestamps.extend(df.index)
#             timestamps_dict[key] = df.index

#         return sorted(set(all_timestamps)), timestamps_dict
################################################################################





################################################################################
def get_df_dict_timestamps_from_nested_dict(dict_of_device_registers_dfs_dicts: dict[str, dict[str, pd.DataFrame | xr.DataArray]],
                                            type_key: str | None = None,
                                            devices: list[str] | None = None,
                                            channel_type_registerIDs: list[str] | None = None,
                                            return_type: str = 'list', # 'dict' or 'list' or 'both'
                                            verbose: bool = True,
                                            ) -> dict[str, pd.Series] | list[pd.Series] | tuple[dict[str, pd.Series], list[pd.Series]]:
    '''
    Wrapper for collect_timestamps_nested_dict function.

    Parameters:
        dict_of_device_registers_dfs_dicts (dict): device_dfs_dict {device: {register_address: DataFrame(Time index, localID columns)}}
        type_key (str): optional peripheral type to use (e.g., the global type_key) if using channel_type_registerIDs
        devices (list or None): optional subset of devices to include; defaults to all in device_dfs_dict
        channel_type_registerIDs (list or None): optional subset of peripheral register IDs to include; defaults to all for each device
        return_type (str): 'list', 'dict', or 'both' (passed through to collect_timestamps_dict)
        verbose (bool): If True, prints warnings and information during processing.
    
    Returns:
        list | dict | (list, dict): timestamps per return_type
    '''
    print("Warning: 'get_df_dict_timestamps_from_nested_dict' is deprecated. Please use 'collect_timestamps_nested_dict' instead.")
    return collect_timestamps_nested_dict(device_dfs_dict=dict_of_device_registers_dfs_dicts,
                                          data_key=type_key,
                                          devices=devices,
                                          peripheral_registerIDs=channel_type_registerIDs,
                                          return_type=return_type,
                                          verbose=verbose
                                          )

##--------------------------------------------------------------------------------   Original
# def get_df_dict_timestamps_from_nested_dict(
#         dict_of_devices_registers_dfs_dicts=None,
#         type_key=None,
#         devices=None,
#         return_type='list'
# ):
#         """
#         Collect timestamps across all devices/registers relevant to the given type_key from the nested dict_of_devices_registers_dfs_dicts using get_df_dict_timestamps.

#         Parameters:
#             dict_of_devices_registers_dfs_dicts (dict): dict_of_devices_registers_dfs_dicts {device: {register_address: DataFrame(Time index, localID columns)}}
#             type_key (str): channel type to use (e.g., the global type_key)
#             devices (list or None): optional subset of devices to include; defaults to all in dict_of_devices_registers_dfs_dicts
#             return_type (str): 'list', 'dict', or 'both' (passed through to get_df_dict_timestamps)

#         Returns:
#             list | dict | (list, dict): timestamps per return_type
#         """
#         if not isinstance(dict_of_devices_registers_dfs_dicts, dict) or not dict_of_devices_registers_dfs_dicts:
#                 print("Warning: dict_of_devices_registers_dfs_dicts is empty or not a dict.")
#                 return [] if return_type == 'list' else ({} if return_type == 'dict' else ([], {}))

#         # Use provided devices or all devices present in the nested dict
#         devices = devices if devices is not None else list(dict_of_devices_registers_dfs_dicts.keys())

#         # Map localID -> register address for this type_key, then get unique register addresses
#         reg_map = channel_type_registerIDs[type_key]  # e.g., {'DIPort0': '32', ...}
#         needed_registers = sorted(set(reg_map.values()))

#         # Build a flat dict of DataFrames to feed into get_df_dict_timestamps
#         # Keyed by "{device}:{register}"
#         df_dict = {}
#         for device in devices:
#                 if device not in dict_of_devices_registers_dfs_dicts:
#                         print(f"Warning: Device {device} not found in dict_of_devices_registers_dfs_dicts; skipping.")
#                         continue

#                 dev_regs = dict_of_devices_registers_dfs_dicts[device]
#                 for reg in needed_registers:
#                         if reg not in dev_regs:
#                                 print(f"Warning: Device {device} missing register {reg}; skipping.")
#                                 continue

#                         # Restrict columns to localIDs relevant to this type_key and register
#                         localIDs_for_reg = [lid for lid, r in reg_map.items() if r == reg and lid in dev_regs[reg].columns]
#                         if not localIDs_for_reg:
#                                 print(f"Warning: Device {device} register {reg} has no expected localIDs present; skipping.")
#                                 continue

#                         df_subset = dev_regs[reg][localIDs_for_reg]
#                         df_dict[f"{device}:{reg}"] = df_subset

#         if not df_dict:
#                 print("Warning: No DataFrames collected for the specified type_key.")
#                 return [] if return_type == 'list' else ({} if return_type == 'dict' else ([], {}))

#         if return_type == 'list':
#                 return get_df_dict_timestamps(df_dict=df_dict, return_type='list')
#         elif return_type == 'dict':
#                 return get_df_dict_timestamps(df_dict=df_dict, return_type='dict')
#         elif return_type == 'both':
#                 return get_df_dict_timestamps(df_dict=df_dict, return_type='both')
#         else:
#                 raise ValueError("return_type must be one of 'list', 'dict', or 'both'.")
################################################################################























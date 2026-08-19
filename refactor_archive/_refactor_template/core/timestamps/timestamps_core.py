'''
Timetamps Module. 
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

from typing import Any

import pandas as pd
import xarray as xr

from data_conduit._refactor_template.core.utils import (
    _apply_level_selectors,
    _flatten_nested_dict,
    _get_nested_dict_depth,
)

################################################################################






################################################################################
# Private Helper Functions
################################################################################

#===============================================================================
# 1| Find Columns (Case-Insensitive)
#===============================================================================
def _find_column(columns, name):
    """Return the actual column name matching `name` (case-insensitive), or None."""
    for col in columns:
        if col.lower() == name.lower():
            return col
    return None


#===============================================================================

################################################################################






################################################################################
# Collect Timestamps
################################################################################
def collect_timestamps(df: pd.DataFrame | xr.DataArray | None = None,
                       timestamp_name: str | None = None,
                       time_location: str | None = None,
                       verbose: bool = False
                       ) -> pd.Series | None:
    """
    Collect and process timestamps from a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame | xr.DataArray | None
        The DataFrame or DataArray containing the timestamps.
    timestamp_name : str | None
        The name of the column containing the timestamps. If None, defaults to 'Time'. Note: case-insensitive.
    time_location : str | None
        The location of the timestamps, either 'columns' or 'index'. If None, will attempt index first, then columns.
    verbose : bool
        If True, prints warnings and information during processing.

    Returns
    -------
    pd.Series | None
        A Series of processed timestamps, or None if the DataFrame is None.
    """

    #=== i| Checks
    if df is None:
        if verbose:
            print("No DataFrame provided. Returning None.")
        return None
    
    if isinstance(df, xr.DataArray):                   # Type check, convert to DataFrame if it's an xarray DataArray
        df = df.to_dataframe().reset_index()                         
    elif not isinstance(df, pd.DataFrame):
        raise TypeError(f"Error in function collect_timestamps: Expected df to be a pandas DataFrame or xarray DataArray, but got {type(df)}")
    
    #=== ii| Set default timestamp column
    if not isinstance(timestamp_name, str):
        timestamp_name = 'Time'  # Default column name for timestamps
    

    #=== iii| Extract Data
    ''' 
    Attempt to find the timestamp column in the specified location (index or columns). 
    If time_location is None, will check index first, then columns. 
    '''

    if time_location == 'index':
        
        index_name = df.index.name
        
        if index_name is not None and index_name.lower() == timestamp_name.lower():
            return df.index.to_series()
        else:
            raise ValueError(
                            f'''Error in function collect_timestamps: Expected index name to be "{timestamp_name}" 
                            (case-insensitive), but got "{index_name}".'''
                            )
    
    if time_location == 'columns':
        
        actual_col = _find_column(df.columns, timestamp_name)
        if actual_col is not None:
            if not isinstance(df[actual_col], pd.Series):
                raise TypeError(
                                f'''Error in function collect_timestamps: Expected column '{actual_col}' 
                                to be a pandas Series, but got {type(df[actual_col])}
                                '''
                                )
            return df[actual_col]
    

    if time_location is None:

        # Check index first
        if df.index.name is not None and df.index.name.lower() == timestamp_name.lower():
            return df.index.to_series()
        
        # Check Columns
        actual_col = _find_column(df.columns, timestamp_name)
        if actual_col is not None:
            if not isinstance(df[actual_col], pd.Series):
                raise TypeError(
                                f'''Error in function collect_timestamps: Expected column '{actual_col}' 
                                to be a pandas Series, but got {type(df[actual_col])}
                                '''
                                )
            return df[actual_col]
        else:
            raise ValueError(
                            f'''Error in function collect_timestamps: Could not find a column or index named "{timestamp_name}" 
                            (case-insensitive) in the DataFrame.'''
                            )
    
    raise ValueError(
                    f'''Error in function collect_timestamps: Invalid time_location "{time_location}". 
                    Expected "index", "columns", or None.'''
                    )

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
        dfs (dict): 
            A dictionary where keys are identifiers and values are pandas DataFrames.
        timestamp_name (str, optional): 
            The name of the column containing the timestamps. If None, defaults to 'Time'. Note: case-insensitive.
        verbose (bool, optional): 
            If True, prints warnings and information during processing. Default is True.
        return_type (str, optional): 
            If 'list', returns a sorted list of all timestamps. 
            If 'dict', returns a dictionary of timestamps per DataFrame. Default is 'list'. 
            If 'both', returns two outputs: a sorted list of all timestamps and a dictionary of timestamps per DataFrame.

    Returns:
        list of all timestamps from each DataFrame in the dictionary, sorted in ascending order.
        dict where keys are the same as df_dict and values are the timestamps from each DataFrame.
    """ 

    #=== i| Checks
    if not isinstance(dfs, dict):
        raise TypeError(f"Error in function collect_timestamps_dict: Expected dfs to be a dictionary, but got {type(dfs)}")
    
    # Standardise return_type to lowercase for case-insensitive comparison
    return_type = return_type.lower() 
    if return_type not in ['list', 'dict', 'both']:
    
        raise ValueError(
                        f'''
                         Error in function collect_timestamps_dict: Invalid return_type '{return_type}'. 
                         Expected 'list', 'dict', or 'both'.
                         '''
                         )
    
    #=== ii| Collect Timestamps
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
    
    #=== iii| Return Results
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
# Collect Timestamps from Nested Dictionary (Arbitrary Depth, Generic Function)
################################################################################

def collect_timestamps_nested(dfs_dict: dict[str, Any],
                              return_type: str = 'list',
                              separator: str = ':',
                              verbose: bool = False,
                              **kwargs,                 # l0_selector, l1_selector, l2_selector, etc. for future extensibility
                              ) -> (  # TODO: Refine return type hint to be more specific about the nested structure
        dict[str, pd.Series] | list[pd.Series] | tuple[dict[str, pd.Series], list[pd.Series]]
):
    '''
    Collect timestamps from an arbitrarily nested dictionary of DataFrames/DataArrays.
    
    This function first flattens the nested dictionary structure, then applies the base 
    timestamp collection logic to the flattened dictionary.

    ----------
    Parameters:
        dfs_dict (dict): 
            An arbitrarily nested dictionary where the leaf nodes are pandas DataFrames or xarray DataArrays.
        return_type (str): 
            Determines the format of the returned timestamps. 
            - 'list': Returns a sorted list of all timestamps across all DataFrames.
            - 'dict': Returns a dictionary mapping each flattened key to its corresponding timestamps.
            - 'both': Returns a tuple containing both the dictionary and the sorted list of timestamps.
        separator (str): 
            The separator used to concatenate keys when flattening the nested dictionary. Default is ':'.
        verbose (bool): 
            If True, prints warnings and information during processing. Default is False.
        **kwargs:
            Optional keyword arguments for filtering or selecting specific subsets of the nested dictionary. 
            Must follow naming convention of l{n}_selector to correspond to the depth level n in the nested structure 
            (e.g., l0_selector for top-level keys, l1_selector for second-level keys, etc.).
            Format should be a dictionary where keys are selector levels and values are lists of keys to include at that level.
            e.g. l0_selector = ['device1', 'device2'], l1_selector = ['registerA', 'registerB'], etc.


    Returns:
        Depending on the value of return_type:
        - If 'list': A sorted list of all timestamps collected from the DataFrames.
        - If 'dict': A dictionary where keys are the flattened paths of the original nested keys, and values are the corresponding timestamps.
        - If 'both': A tuple containing both the dictionary and the sorted list of timestamps.
    '''
    #=== i| Checks
    if not isinstance(dfs_dict, dict) or not dfs_dict:
        if verbose:
            print("Warning: Input is not a valid non-empty dictionary.")
        return [] if return_type == 'list' else ({} if return_type == 'dict' else ({}, []))
    if return_type not in ('list', 'dict', 'both'):
        raise ValueError("return_type must be one of 'list', 'dict', or 'both'.")
    
    # Check all keyword arguments have the correct naming convention
    for key in kwargs:
        if not key.startswith('l') or not key.endswith('_selector'):
            raise ValueError(
                f"Invalid keyword argument '{key}'. "
                "All selectors must follow the naming convention "
                "'l{n}_selector' (e.g., l0_selector, l1_selector, etc.)."
            )

    # Check that depth of selectors matches the depth of the nested dictionary
    max_depth = _get_nested_dict_depth(dfs_dict)
    selector_levels = [key for key in kwargs if key.startswith('l') and key.endswith('_selector')]
    for selector in selector_levels:
        level_num = int(selector[1:-9])  # Extract the level number from the selector name
        if level_num >= max_depth:
            raise ValueError(
                f"Selector '{selector}' is defined for level {level_num}, "
                f"but the nested dictionary only has a maximum depth of {max_depth}."
            )

    #=== ii| Apply level selectors before flattening
    if kwargs:
        level_selectors = {
            int(k[1:-9]): (v if isinstance(v, list) else [v])
            for k, v in kwargs.items()
        }
        dfs_dict = _apply_level_selectors(dfs_dict, level_selectors)

    #=== iii| Flatten the nested dictionary
    flat_dict = _flatten_nested_dict(dfs_dict, separator=separator)

    if not flat_dict:
        if verbose:
            print("Warning: No DataFrames collected from the nested structure.")
        return [] if return_type == 'list' else ({} if return_type == 'dict' else ({}, []))
    
    #=== iii| Collect timestamps from the flattened dictionary
    return collect_timestamps_dict(dfs=flat_dict, return_type=return_type, verbose=verbose)

################################################################################







# ################################################################################
# # 3. Domain-Specific Wrapper
# ################################################################################

# def collect_timestamps_nested_dict(dfs_dict: dict[str, dict[str, pd.DataFrame | xr.DataArray]],
#                                    data_key: str | None = None,
#                                    devices: list[str] | None = None,
#                                    peripheral_registerIDs: dict[str, dict[str, str]] | None = None,
#                                    return_type: str = 'list',
#                                    verbose: bool = True) -> dict[str, pd.Series] | list[pd.Series] | tuple[dict[str, pd.Series], list[pd.Series]]:
#     '''
#     Domain-specific wrapper that filters devices and registers based on peripheral_registerIDs,
#     then uses the generic timestamp collector.
#     '''
#     if not isinstance(dfs_dict, dict) or not dfs_dict:
#         if verbose:
#             print("Warning: dfs_dict is not a valid non-empty dictionary.")
#         return [] if return_type == 'list' else ({} if return_type == 'dict' else ({}, []))

#     target_devices = devices if devices is not None else list(dfs_dict.keys())
#     filtered_dfs = {}

#     if peripheral_registerIDs is not None and data_key is not None:
#         reg_map = peripheral_registerIDs.get(data_key, {})
#         needed_registers = sorted(set(reg_map.values()))

#         for device in target_devices:
#             if device not in dfs_dict:
#                 if verbose: 
#                     print(f"Warning: Device {device} not found in dfs_dict; skipping.")
#                 continue

#             dev_regs = dfs_dict[device]
#             for reg in needed_registers:
#                 if reg not in dev_regs:
#                     if verbose: 
#                         print(f"Warning: Device {device} missing register {reg}; skipping.")
#                     continue

#                 # Filter columns based on localIDs
#                 localIDs_for_reg = [lid for lid, r in reg_map.items() if r == reg and lid in dev_regs[reg].columns]
#                 if not localIDs_for_reg:
#                     if verbose: 
#                         print(f"Warning: Device {device} register {reg} has no expected localIDs present; skipping.")
#                     continue

#                 # Build the filtered nested structure
#                 if device not in filtered_dfs:
#                     filtered_dfs[device] = {}
#                 filtered_dfs[device][reg] = dev_regs[reg][localIDs_for_reg]

#     elif peripheral_registerIDs is None and data_key is None:
#         for device in target_devices:
#             if device not in dfs_dict:
#                 if verbose: 
#                     print(f"Warning: Device {device} not found in dfs_dict; skipping.")
#                 continue
#             filtered_dfs[device] = dfs_dict[device]
            
#     else:
#         raise ValueError("peripheral_registerIDs and data_key must be provided together, or both left as None.")

#     # Hand off the cleanly filtered nested dictionary to the generic function
#     return collect_timestamps_nested(
#         dfs_dict=filtered_dfs,
#         return_type=return_type,
#         separator=':',
#         verbose=verbose
#    )




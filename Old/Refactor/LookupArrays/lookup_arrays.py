'''
Utilities for Creating Lookup Arrays with Virtual Coordinates and Custom Accessors

Functions:
-------------------------
- create_lookup_array: Create a lookup array with virtual coordinates and custom accessors.
    - Should allow for arbitrary number of virtual coordinate dimensions.

- construct_base_da: Construct a base DataArray with specified dimensions and coordinates.
    - Base version of construct_channel_type_da with more general applicability.

- 
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

from collections.abc import Mapping

################################################################################


################################################################################
# Construct Base DataArray
################################################################################
def construct_base_da(values: list | np.ndarray | None= None,
                      unified_dim: list = None,
                      unified_coord_name: str= 'unified_coord',
                      virtual_coord_names: list= None,
                      virtual_map_dict: dict= None,
                      dict_of: str= 'dicts',
                      times: list | np.ndarray= None,
                      name: str= 'base_dataarray',
                      test_help_values: bool= False,
                      verbose: bool= False
                      ) -> xr.DataArray:
    '''
    Construct a base/placeholder DataArray for a set of given unified/global coordinates for all timepoints and their associated virtual coordinates.
    -----------------------------------

    Parameters:
        values (list or np.ndarray): Values to populate the DataArray. If None, a placeholder array of NaNs will be created.
        unified_dim (list): List of unified/global coordinate names to include in the DataArray.
        virtual_coord_names (list): List of virtual coordinate dimensions to include in the DataArray.
        virtual_map_dict (dict): Dictionary mapping unified/global coordinates to their virtual coordinate information.
        dict_of (str): Specifies the format of virtual_map_dict. Options are 'dicts' or 'tuples'.
        times (list): List of timepoints to include in the DataArray. This should be obtained using collect_timestamps_dict with return_type='list'.
        name (str): Name of the DataArray.
        test_help_values (bool): If True, use test helper values instead of the main values.
        verbose (bool): If True, print verbose output.
    
    Returns:
        A DataArray populated with the specified values, or a placeholder array if no values are provided.
    '''
    

    dict_of= dict_of.lower()

    if dict_of not in ("dicts", "tuples"):
        raise ValueError(f"dict_of must be either 'dicts' or 'tuples', got: {dict_of}")

    if times is None or unified_dim is None or virtual_coord_names is None or virtual_map_dict is None:
        raise ValueError("times, unified_dim, virtual_coord_names, and virtual_map_dict must be provided.")
    
    missing= [u for u in unified_dim if u not in virtual_map_dict]
    if missing: raise KeyError(f'Error: virtual_map_dict is missing entries for unified coordinates: {missing[:5]}{"..." if len(missing) > 5 else ""}')
    
    if dict_of == "tuples" and any(isinstance(v, dict) for v in virtual_map_dict.values()):
        raise TypeError("dict_of='tuples' but at least one entry is a dict. Use dict_of='dicts'.")

    if dict_of == "dicts" and any(isinstance(v, (tuple, list)) for v in virtual_map_dict.values()):
        raise TypeError("dict_of='dicts' but at least one entry is a tuple/list. Use dict_of='tuples'.")



    #=== Create data to populate DataArray 
    if values is None and test_help_values is False:
        if verbose: print(f'No values provided, test_help_values is False. Creating NaNs placeholder DataArray with {len(times)} timestamps and {len(unified_dim)} unified coordinates.')
        values= np.zeros((len(times), len(unified_dim)), dtype= bool) 

    #=== Create test helper values if specified
    if values is None and test_help_values is True:
        if verbose: print(f'No values provided, but test_help_values is True. Creating test helper DataArray with {len(times)} timestamps and {len(unified_dim)} unified coordinates.')
        value_list = []
        for _ in times:
            time_values = []
            for unified in unified_dim:
                virtual_coords = virtual_map_dict[unified]

                if dict_of == 'dicts':
                    if not isinstance(virtual_coords, Mapping): raise TypeError(f'Expected virtual_map_dict to be dict-of-dicts when dict_of="dicts", but got {type(virtual_coords)} for unified coordinate {unified}')
                    
                    coord_str = ', '.join([f'{k}: {v}' for k, v in virtual_coords.items()])

                elif dict_of == 'tuples':
                    
                    if not isinstance(virtual_coords, (tuple, list)): raise TypeError(f'Expected virtual_map_dict to be dict-of-tuples/lists when dict_of="tuples", but got {type(virtual_coords)} for unified coordinate {unified}')
                    
                    if len(virtual_coords) != len(virtual_coord_names): raise ValueError(f"Tuple length mismatch for {unified}: len(entry)={len(virtual_coords)} vs len(virtual_coord_names)={len(virtual_coord_names)}")
                    
                    coord_str = ", ".join([f"{k}: {v}" for k, v in zip(virtual_coord_names, virtual_coords)])

                else: raise TypeError(f"dict_of parameter must be either 'dicts' or 'tuples', got: {dict_of}")

                time_values.append(f'({unified}, {coord_str})')

            value_list.append(time_values)
        values = np.array(value_list)
        if verbose: print(f'Constructed test helper values array with shape: {values.shape}')
    
    if values is not None and test_help_values is True:
        print('Warning: Both values provided and test_help_values is True. Using provided values.')
    
    #=== Build coordinates 
    
    # Primary/Base coordinates
    coords= {
        'Time': times,
        unified_coord_name: unified_dim,
        }
    
    # Virtual coordinates (dict-of-dicts / named fields)
    if dict_of == 'dicts':
        for vcoord in virtual_coord_names:
            try: vcoord_values= [virtual_map_dict[unified][vcoord] for unified in unified_dim]
            except KeyError as e:
                raise KeyError(f'Missing key {e} in virtual_map_dict entries (dict_of="dicts")') from e
            
            coords[vcoord] = (unified_coord_name, vcoord_values)
    
    
    # Virtual coordinates (dict-of-tuples / positional fields)
    elif dict_of == 'tuples':
        for i, vcoord in enumerate(virtual_coord_names):
            try: vcoord_values = [virtual_map_dict[unified][i] for unified in unified_dim]
            except IndexError as e:
                raise IndexError(f'Index {i} out of range for virtual_map_dict entries (dict_of="tuples")') from e
            
            coords[vcoord] = (unified_coord_name, vcoord_values)
    
    else: raise TypeError(f"dict_of parameter must be either 'dicts' or 'tuples', got: {dict_of}")

    ########################################################
    """
    NOTE: virtual_map_dict can be either:
      (A) dict-of-dicts (recommended):
          virtual_map_dict[unified] = {"device": ..., "localID": ..., ...}
          -> use key lookup: virtual_map_dict[unified][vcoord]
    
      (B) dict-of-tuples/lists (positional):
          virtual_map_dict[unified] = (device, localID, ...)
          -> assumes tuple order matches virtual_coord_names
    
    Example for (B):
    for i, vcoord in enumerate(virtual_coord_names):
        vcoord_values = [virtual_map_dict[unified][i] for unified in unified_dim]
        coords[vcoord] = (unified_coord_name, vcoord_values)
    
    Dict-of-tuples requires that the order of elements in the tuple matches the order in virtual_coord_names.
    """
    ########################################################

    #=== Create DataArray
    da= xr.DataArray(
        data=values,
        dims= ['Time', unified_coord_name],
        coords= coords,
        name= name,
        attrs= {
            'description': f'Base DataArray for unified coordinates of type {name}',
            'source': 'Constructed using construct_base_da function'
        }
    )
    return da
################################################################################



################################################################################
# Construct Lookup Array
################################################################################
def construct_lookup_array(unified_dim: list= None,
                           unified_coord_name: str= 'unified_coord',
                           virtual_coord_names: list= None,
                           virtual_map_dict: dict= None,
                           dict_of: str= 'dicts',
                           name: str= 'lookup_matrix',
                           verbose: bool= False
                           ) -> tuple[xr.DataArray, dict[str, list]]:
    '''
    Construct a Lookup Array DataArray for a set of given unified/global coordinates and their associated virtual coordinates.
    -----------------------------------

    Parameters:
        unified_dim (list): List of unified/global coordinate names to include in the DataArray.
        unified_coord_name (str): Name for the unified/global coordinate dimension.
        virtual_coord_names (list): List of virtual coordinate dimensions to include in the DataArray.
        virtual_map_dict (dict): Dictionary mapping unified/global coordinates to their virtual coordinate information.
        dict_of (str): Specifies the format of virtual_map_dict. Options are 'dicts' or 'tuples'.
        name (str): Name of the DataArray.
        verbose (bool): If True, prints additional information. 
    
    Returns:
        lookup_da (xr.DataArray): A DataArray representing the lookup matrix.
        lookup_virtual_coords (dict): A dictionary containing lists of unique virtual coordinate values for each virtual coordinate dimension. Keys are virtual coordinate names, values are lists of unique values. 
    '''
    #==== Input validation and preprocessing ====
    dict_of= dict_of.lower()

    if dict_of not in ("dicts", "tuples"):
        raise ValueError(f"dict_of must be either 'dicts' or 'tuples', got: {dict_of}")
    
    if unified_dim is None or virtual_coord_names is None or virtual_map_dict is None:
        raise ValueError("unified_dim, virtual_coord_names, and virtual_map_dict must be provided.")
    
    if len(unified_dim) == 0:
        raise ValueError("unified_dim cannot be empty.")
    if len(virtual_coord_names) == 0:
        raise ValueError("virtual_coord_names cannot be empty.")
    
    if verbose: print(f'Constructing lookup array for {len(unified_dim)} unified coordinates and virtual coordinates: {virtual_coord_names}')
    
    missing= [u for u in unified_dim if u not in virtual_map_dict]
    if missing: raise KeyError(f'Error: virtual_map_dict is missing entries for unified coordinates: {missing[:5]}{"..." if len(missing) > 5 else ""}')
    
    if dict_of == "tuples" and any(isinstance(v, dict) for v in virtual_map_dict.values()):
        raise TypeError("dict_of='tuples' but at least one entry is a dict. Use dict_of='dicts'.")

    if dict_of == "dicts" and any(isinstance(v, (tuple, list)) for v in virtual_map_dict.values()):
        raise TypeError("dict_of='dicts' but at least one entry is a tuple/list. Use dict_of='tuples'.")

    #==== Collect Unique Values for Each Virtual Coordinate (Levels)
    lookup_virtual_coords= {vcoord: [] for vcoord in virtual_coord_names}

    for unified in unified_dim:
        virtual_coords= virtual_map_dict[unified]

        if dict_of == 'tuples':
            if not isinstance(virtual_coords, (tuple, list)):
                raise TypeError(f'Expected virtual_map_dict to be dict-of-tuples/lists when dict_of="tuples", but got {type(virtual_coords)} for unified coordinate {unified}')
            if len(virtual_coords) != len(virtual_coord_names):
                raise ValueError(f"Tuple length mismatch for {unified}: len(entry)={len(virtual_coords)} vs len(virtual_coord_names)={len(virtual_coord_names)}")
            
            for vcoord_name, vcoord_value in zip(virtual_coord_names, virtual_coords):
                if vcoord_value not in lookup_virtual_coords[vcoord_name]:
                    lookup_virtual_coords[vcoord_name].append(vcoord_value)

        elif dict_of == 'dicts':
            if not isinstance(virtual_coords, Mapping):
                raise TypeError(f'Expected virtual_map_dict to be dict-of-dicts when dict_of="dicts", but got {type(virtual_coords)} for unified coordinate {unified}')
            
            for vcoord_name in virtual_coord_names:
                if vcoord_name not in virtual_coords:
                    raise KeyError(f'Missing key {vcoord_name} in virtual_map_dict entry for unified coordinate {unified} (dict_of="dicts")')
                
                vcoord_value= virtual_coords[vcoord_name]
                if vcoord_value not in lookup_virtual_coords[vcoord_name]:
                    lookup_virtual_coords[vcoord_name].append(vcoord_value)
        
        else: raise TypeError(f"dict_of parameter must be either 'dicts' or 'tuples', got: {dict_of}")

        if verbose: 
            for vcoord_name in virtual_coord_names: 
                print(f'Collected {vcoord_name}: {lookup_virtual_coords[vcoord_name]} unique values.')
        

    #==== Construct Lookup Array ====
    
    # Fast lookup tables from virtual coordinate values to integer indices along each virtual dimension
    index_map= {
        vcoord: {val: idx for idx, val in enumerate(lookup_virtual_coords[vcoord])} 
        for vcoord in virtual_coord_names
        }

    # Allocate boolean array for lookup matrix
    lookup_shape= [len(unified_dim)] + [len(lookup_virtual_coords[vcoord]) for vcoord in virtual_coord_names]
    lookup_arr= np.zeros(lookup_shape, dtype= bool)

    # Populate lookup array
    for i, unified in enumerate(unified_dim):
        
        entry= virtual_map_dict[unified]

        if dict_of == 'tuples':
            idx= [i] + [index_map[vcoord_name][vcoord_value] 
            for vcoord_name, vcoord_value in zip(virtual_coord_names, entry)
            ]
        
        elif dict_of == 'dicts':
            idx= [i] + [index_map[vcoord_name][entry[vcoord_name]]
            for vcoord_name in virtual_coord_names
            ]
        else: raise TypeError(f"dict_of parameter must be either 'dicts' or 'tuples', got: {dict_of}")
        lookup_arr[tuple(idx)] = True
    
    #==== Create DataArray 
    dims= [unified_coord_name] + virtual_coord_names
    
    coords= {unified_coord_name: unified_dim,}

    coords.update({vcoord: lookup_virtual_coords[vcoord] for vcoord in virtual_coord_names})

    lookup_da= xr.DataArray(
        data= lookup_arr,
        dims= dims,
        coords= coords,
        name= name,
        attrs= {
            'description': 'Lookup matrix indicating presence of unified coordinates across virtual coordinate dimensions',
            'source': 'Constructed using construct_lookup_array function'
        }
    )
    return lookup_da, lookup_virtual_coords

################################################################################




################################################################################
# Select Unified/Global Coordinates from Lookup Array
################################################################################
def ulookup(lookup_arr: xr.DataArray = None,
            unified_coord_name: str= 'unified_coord',
            verbose: bool= False,
            **selectors,
        ) -> list:
    '''
    uLookup: Selects unified/global coordinates from a lookup DataArray based on specified criteria across virtual coordinate dimensions. Supports both positional- and label-based indexing and combinations thereof.
    -----------------------------------

    Parameters:
        lookup_arr (xr.DataArray): N-dim Boolean lookup tensor with dimensions like [unified_coord_name] + virtual coordinate dimensions.
        unified_coord_name (str): Name of the unified/global coordinate dimension.
        verbose (bool): If True, prints additional information.
        **selectors: Keyword arguments specifying selection criteria for both real and virtual coordinate dimensions. Keys are  coordinate names, values are the desired values to filter by. 
    
    Returns:
        list: A list of unified/global coordinate names that match the specified criteria.
    '''
    if lookup_arr is None:
        raise ValueError("lookup_arr must be provided.")
    if unified_coord_name not in lookup_arr.dims:
        raise ValueError(f"{unified_coord_name} is not a dimension in the provided lookup_arr.")
    if verbose: print(f'Starting uLookup with unified_coord_name="{unified_coord_name}" and selectors: {selectors}')


    #=== Gather selection criteria
    sel= {}

    sel= {key: value for key, value in selectors.items() if value is not None}
    if verbose: print(f'Selecting unified coordinates with criteria: {sel}')

    #=== Get subset based on selection criteria
    subset= lookup_arr.sel(**sel) if sel else lookup_arr
    if verbose: print(f'Subset shape after selection: {subset.shape}')

    #=== Reduce across non-unified_coord_name dimensions
    other_dims= [d for d in subset.dims if d != unified_coord_name]
    
    if other_dims:
        per_unified_true= subset.any(dim= other_dims)
    else:
        per_unified_true= subset.astype(bool)
    
    if verbose: print(f'Per-unified_coord_name true shape after reduction: {per_unified_true.shape}')
    
    # return per_unified_true[unified_coord_name].values[per_unified_true.values].tolist()

    labels = per_unified_true[unified_coord_name].values
    mask = np.asarray(per_unified_true.values, dtype=bool)
    return labels[mask].tolist()

################################################################################




################################################################################
# Lookup Accessor with Auto-Construction of Lookup Array
################################################################################
@xr.register_dataarray_accessor('ul')
class LookupAccessorConstructor:
    '''
    Base class for making custom xarray DataArray accessors for lookup operations with auto-construction of lookup arrays.
    '''

    def __init__(self,
                 data_array: xr.DataArray
                 ):
        
        self._da= data_array

        lookup_array= self.lookup_constructor(lookup_array= None,
                                             unified_coord_name= None,
                                             dict_of= None,
                                             verbose= False,
                                             )
        self.lookup_array= lookup_array



    def lookup_constructor(self,
                           lookup_array: xr.DataArray | None= None,
                           unified_coord_name: str | None = None,
                           dict_of: str | None= None,
                           verbose: bool= False,
                           ):
        
        if lookup_array: 
            self.lookup_array= lookup_array
            if verbose: print(f'Using provided lookup_array with shape {self.lookup_array.shape}.')

        if dict_of is None:
            if verbose: print('No preferred format for "dict_of" provided, defaulting to "tuples".')
            dict_of= 'tuples'
        elif dict_of.lower() not in ('tuples', 'dicts'):
            raise ValueError(f'Invalid dict_of value: {dict_of}. Must be either "tuples" or "dicts".')
        
        if not hasattr(self, 'lookup_array'):
            
            if verbose: print(f'Lookup array not provided, attempting to auto-construct using data array')

            #==== 1) Unified Coordinate Name
            """
            Name of unified/global coordinate dimension in the lookup array. Assumed to be second dimension in the data array, or first dimension not including 'Time' dimension.
            """
            if unified_coord_name is None:
                unified_coord_name= [dim for dim in self._da.dims if dim.lower() != 'time'][0]
                if verbose: print(f'No unified_coord_name provided, inferred as "{unified_coord_name}" from data array dimensions.')
            
            #==== 2) Unified Dimension Values
            """
            List of unified/global coordinate names to include in the DataArray.
            """
            unified_dim = self._da.coords[unified_coord_name].values.tolist()
            if verbose: print(f'Extracted {len(unified_dim)} unified coordinates from data array dimension "{unified_coord_name}".')

            #==== 3) Virtual Coordinate Names
            """
            Names of coordinates that map each unified/global coordinate to its virtual coordinate information.
            Assumed to be all coordinates in the data array except for the unified coordinate and 'Time' dimension.
            """
            virtual_coord_names= [coord for coord in self._da.coords.keys() if coord != unified_coord_name and coord.lower() != 'time']
            if verbose: print(f'Inferred virtual coordinate names: {virtual_coord_names}.')

            #==== 4) Virtual Map Dictionary
            """
            Dictionary mapping unified/global coordinates to their virtual coordinate information. Can be in dict-of-dicts or dict-of-tuples format.
            """
            if dict_of == 'tuples':
                # Fix: Use dict() to create proper dictionary from zip
                virtual_map_dict = dict(zip(
                    unified_dim, 
                    zip(*(self._da.coords[coord].values.tolist() for coord in virtual_coord_names))
                ))
            elif dict_of == 'dicts':
                virtual_map_dict = {
                    unified: {vcoord: self._da.sel({unified_coord_name: unified})[vcoord].item() for vcoord in virtual_coord_names}
                    for unified in unified_dim
                }
            if verbose: print(f'Constructed virtual_map_dict with {len(virtual_map_dict)} entries in format "{dict_of}".')

            #==== 5) Construct Lookup Array
            self.lookup_array, _ = construct_lookup_array(
                unified_dim= unified_dim,
                unified_coord_name= unified_coord_name,
                virtual_coord_names= virtual_coord_names,
                virtual_map_dict= virtual_map_dict,
                dict_of= dict_of,
                name= f'{unified_coord_name}_lookup_matrix',
                verbose= verbose
            )
            if verbose: print(f'Constructed lookup_array with shape {self.lookup_array.shape}.')
        return self.lookup_array
    
    def ul_sel(self,
                verbose: bool= False,
                **selectors,
            ) -> list:
        '''
        uLookup: Selects unified/global coordinates from the lookup DataArray based on specified criteria across virtual coordinate dimensions. Supports both positional- and label-based indexing and combinations thereof.
        -----------------------------------

        Parameters:
            verbose (bool): If True, prints additional information.
            **selectors: Keyword arguments specifying selection criteria for both real and virtual coordinate dimensions. Keys are  coordinate names, values are the desired values to filter by.
        Returns:
            list: A list of unified/global coordinate names that match the specified criteria.
        '''
        u_c_n= [dim for dim in self._da.dims if dim.lower() != 'time'][0]
        return ulookup(
            lookup_arr= self.lookup_array,
            unified_coord_name= u_c_n,
            verbose= verbose,
            **selectors
        )
    
    def sel(self, **selectors) -> list:
        '''
        sel: Alias for ulookup method to select unified/global coordinates based on specified criteria.
        -----------------------------------

        Parameters:
            **selectors: Keyword arguments specifying selection criteria for both real and virtual coordinate dimensions. Keys are  coordinate names, values are the desired values to filter by.
        Returns:
            list: A list of unified/global coordinate names that match the specified criteria.
        '''
        return self.ul_sel(**selectors)
    
    __call__= sel



################################################################################




################################################################################
# Collect Timestamps
################################################################################
@xr.register_dataarray_accessor('cdl')
class CDLAccessor:
    """
    This accessor adds methods to xarray DataArrays to select channels using a
    channel-device-localID lookup matrix.
    """

    #===== Constructor =====
    def __init__(self, da: xr.DataArray):
        self._da = da

        # Checks
        if "channel" not in self._da.dims:
            raise ValueError("DataArray must have 'channel' as one of its dimensions.")
        for coord in ["device", "localID"]:
            if coord not in self._da.coords:
                raise ValueError(f"DataArray must have '{coord}' as one of its coordinates.")

        #--- Create lookup DataArray
        channels = da.channel.values.tolist()

        devices = list(dict.fromkeys(da.device.values.tolist()))
        localIDs = list(dict.fromkeys(da.localID.values.tolist()))

        lookup_arr = np.zeros((len(channels), len(devices), len(localIDs)), dtype=bool)

        for i, (d, l) in enumerate(zip(da.device.values, da.localID.values)):
            device_index = devices.index(d)
            localID_index = localIDs.index(l)
            lookup_arr[i, device_index, localID_index] = True

        self.lookup_da = xr.DataArray(
            data=lookup_arr,
            dims=["channel", "device", "localID"],
            coords={
                "channel": channels,
                "device": devices,
                "localID": localIDs,
            },
            name="channel_lookup_matrix",
            attrs={
                "description": "Channel lookup matrix indicating presence of channels across devices and local IDs",
                "source": "Constructed using CDLAccessor",
            },
        )

    #===== Methods =====
    def select_channels_from_C_D_L(self, channel=None, device=None, localID=None):
        """
        Select channels from the Channel-Device-LocalID lookup DataArray.
        """

        sel = {}
        if channel is not None:
            sel["channel"] = channel
        if device is not None:
            sel["device"] = device
        if localID is not None:
            sel["localID"] = localID

        # Subset
        subset = self.lookup_da.sel(**sel) if sel else self.lookup_da
        print(f"Subset shape after selection: {subset.shape}")

        # Reduce across non-channel dimensions
        other_dims = [d for d in subset.dims if d != "channel"]
        if other_dims:
            per_channel_true = subset.any(dim=other_dims)
        else:
            per_channel_true = subset.astype(bool)

        print(f"Per-channel true shape after reduction: {per_channel_true.shape}")

        return per_channel_true["channel"].values[per_channel_true.values].tolist()

    def sel(self, *, channel=None, device=None, localID=None, **sel_kwargs):
        """
        Wrapper around xarray .sel but using CDL lookup to choose valid channels.
        """
        selected_channels = self.select_channels_from_C_D_L(
            channel=channel, device=device, localID=localID
        )
        return self._da.sel(channel=selected_channels, **sel_kwargs)

    # Make callable equivalent to .sel
    __call__ = sel



################################################################################




################################################################################
# Collect Timestamps
################################################################################
#### 12a) Get Data Array Index for Given Timestamp

def get_da_index_for_timestamp(channel_type_da= None, Time= None, nearest_match_type= 'Exact', precision= None):
    '''
    Find index of a given timestamp in a DataArray for a specified timestamp. 
    -----------------------------------

    Parameters:
        channel_type_da (xr.DataArray): DataArray with Time coordinate/dimension.
        Time (int, str, or pd.Timestamp): The timestamp to find in the DataArray.
        nearest_match_type (str): Type of match to find ('Exact', 'Nearest', 'Before', 'After').
        precision (str): Precision for rounding the timestamp (e.g., '1ms', '1s'). Not currently implemented, hence None.
    
    Returns:
       int: Index of the timestamp in the DataArray.
    '''
    if Time is None:
        raise ValueError("Parameter 'Time' must be provided.")
    
    
    if precision is not None:
        raise NotImplementedError("Precision parameter is not implemented yet.")

    if nearest_match_type not in ['Exact', 'Nearest', 'Before', 'After']:
        raise ValueError("nearest_match_type must be one of 'Exact', 'Nearest', 'Before', 'After'.")
    
    if 'Time' not in channel_type_da.dims and 'Time' not in channel_type_da.coords:
        raise ValueError("DataArray must have 'Time' as one of its dimensions or coordinates.")
    
    try:
        if nearest_match_type == 'Exact':
            index = channel_type_da.get_index('Time').get_loc(Time)
        elif nearest_match_type == 'Nearest':
            index = channel_type_da.get_index('Time').get_indexer([Time], method='nearest')[0]
        elif nearest_match_type == 'Before':
            index = channel_type_da.get_index('Time').get_indexer([Time], method='pad')[0]
        elif nearest_match_type == 'After':
            index = channel_type_da.get_index('Time').get_indexer([Time], method='backfill')[0]
        
        if index == -1:
            raise ValueError(f"No matching timestamp found for {Time} with method '{nearest_match_type}'.")
        
        return index
    
    except KeyError:
        raise ValueError(f"Timestamp {Time} not found in DataArray.")
    
#### 12B) Accessor Method for get_da_index_for_timestamp
@xr.register_dataarray_accessor('time_indexer')
class TimeIndexerAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def get_index_for_timestamp(self, time, nearest_match_type='Exact', precision=None):
        return get_da_index_for_timestamp(self._obj, time, nearest_match_type, precision)

################################################################################




################################# ###############################################
# Collect Timestamps
################################################################################
#### 13a) Update Individual Channel Type DataArray Entry

def update_channel_type_da_entry(channel_type_da= None,     # Channel Type DataArray to update. Structure: _.loc[dict(Time= time,
                                                                # channel= channel_type_da.cdl.select_channels_from_C_D_L(channel= None, device= device, localID= localID))]
                                 time= None,                # Time to update. 
                                 channel= None,             # Optional, not sure why this would be used but included for completeness
                                 device= None,              # Device to update.
                                 localID=None,              # LocalID to update.
                                 reg_address=None,          # Register address to update.
                                 dict_of_devices_registers_dfs_dicts= None,  # Nested dict of DataFrames with actual data. Structure: _.[device][register_address][localID].loc(time)
                                 device_register_dfs_dict= None,             # Dict of dataframes for single register address. Structure: _.[device][localID].loc(time)
                                 ):
    '''
    Update a single entry in the channel type DataArray with actual data from the provided DataFrames.
    -----------------------------------
    Parameters:
        channel_type_da (xr.DataArray): DataArray with channel type information to update.
        time: Time to update.
        channel (str): Channel to update.
        device (str): Device to update.
        localID (str): LocalID to update.
        reg_address (str): Register address to update.
        dict_of_devices_registers_dfs_dicts (dict): Nested dict of DataFrames with actual data.
        device_register_dfs_dict (dict): Dict of dataframes for single register address.
    '''
    if time is None:
        time= dict_of_devices_registers_dfs_dicts[device][reg_address][localID].index[0] if dict_of_devices_registers_dfs_dicts is not None else device_register_dfs_dict[device][localID].index[0]

    if reg_address is None and device_register_dfs_dict is not None:
        channel_type_da.loc[dict(Time= time, channel= channel_type_da.cdl.select_channels_from_C_D_L(channel= None, device= device, localID= localID))] = device_register_dfs_dict[device][localID].loc[time]

    if reg_address is not None and device_register_dfs_dict is None:
        channel_type_da.loc[dict(Time= time, channel= channel_type_da.cdl.select_channels_from_C_D_L(channel= None, device= device, localID= localID))] = dict_of_devices_registers_dfs_dicts[device][reg_address][localID].loc[time]

    return None


#### 13b v2) Update Channel Type DataArray

def update_channel_type_da(
        channel_type_da= None, # Existing DataArray to update
        channel_ref_dict= None,     # {channel: (device, localID)}
        dict_of_devices_registers_dfs_dicts= None, # {device: {register_address: DataFrame(index= Time, columns= localIDs)}}
        device_register_dfs_dict= None, # {device: DataFrame(index= Time, columns= localIDs)}
        channel_type_registerIDs= None, # {type_key: {localID: register_address}}
        type_key= None,
        fill_value= None,
        verbose= False

):
    
    '''
    Update channel_type_da at (Time, channel) positions using: dict_of_devices_registers_dfs_dicts[device][reg_address][localID].   

    Steps per channel:
        Map channel -> (device, localID) via channel_ref_dict
        Map localID -> reg_address via channel_type_registerIDs[type_key][localID]
        Pull source column; align Time to channel_type_da.Time (intersection)
        Optional fill within aligned rows; drop NaNs
        Collect columns into a (Time x channel) patch and assign once

    Parameters:
        channel_type_da (xr.DataArray): Existing DataArray to update.
        channel_ref_dict (dict): Dictionary mapping channel names to their reference information.
        dict_of_devices_registers_dfs_dicts (dict): Nested dictionary of devices and their register DataFrames.
        device_register_dfs_dict (dict or None): Optional pre-constructed dict of device DataFrames to use instead of nested dict.
        channel_type_registerIDs (dict): Mapping of type_key to localID to register_address.
        type_key (str): The type_key corresponding to the channel_type_da.
        fill_value: Value to use for missing data. If None, will use NaN for floats and False for bools.
        verbose (bool): If True, prints detailed output during the update process.

    Returns:
        xr.DataArray: Updated DataArray with new data.
    '''

    #===== Error Checks =====
    if dict_of_devices_registers_dfs_dicts is None:
        raise ValueError("dict_of_devices_registers_dfs_dicts is required for this update path.")
    if channel_type_registerIDs is None or type_key is None:
        raise ValueError("channel_type_registerIDs and type_key are required.")
    if type_key not in channel_type_registerIDs:
        raise ValueError(f"type_key '{type_key}' not found in channel_type_registerIDs.")
    if channel_type_da is None:
        raise ValueError("channel_type_da is required.")
    if channel_ref_dict is None:
        raise ValueError("channel_ref_dict is required.")
    if not isinstance(channel_type_da, xr.DataArray):
        raise ValueError("channel_type_da must be an xarray DataArray.")
    

    #===== Setup =====

    ctda= channel_type_da
    ctda_time= ctda.get_index('Time')
    ctda_channels= ctda.coords['channel'].to_index()

    patch_cols= {} # channel -> aligned pandas column

    #===== Main Loop Over Channels =====

    for channel, (device, localID) in channel_ref_dict.items():

        reg_addr= channel_type_registerIDs[type_key][localID]                                                   # get reg addr for localID for given type_key

        if device not in list(dict_of_devices_registers_dfs_dicts.keys()):
            if verbose:
                print(f'Channel {channel} on device {device} localID {localID} has no device data; skipping.')
            continue

        if reg_addr in list(dict_of_devices_registers_dfs_dicts[device].keys()):
            source_col= dict_of_devices_registers_dfs_dicts[device][reg_addr][localID]                          # Get source column (pd Series-like) indexed by Time
        else:
            if verbose:
                print(f'Channel {channel} on device {device} localID {localID} has no register address {reg_addr}; skipping.')
            continue

        # source_col= dict_of_devices_registers_dfs_dicts[device][reg_addr][localID]  # Get source column (pd Series-like) indexed by Time

        time_index= ctda_time.intersection(source_col.index)                                                    # Aligns Time by intersection with ctda.Time
        if len(time_index) == 0:                                                                                # Skip if no overlap in Time
            if verbose:
                print(f'Channel {channel} on device {device} localID {localID} has no overlapping Time with channel_type_da; skipping.')
            continue

        vals= source_col.reindex(time_index)                                                                    # Re-index to aligned Time, keeps only rows with timestamps present in time_index and with same order
        if fill_value is not None:
            vals= vals.fillna(fill_value)                                                                       # Optional fill within aligned rows
        
        vals= vals.dropna()                                                                                     # Drop NaNs to avoid overwriting with NaN
        if vals.empty:                                                                                          # Skip if no values remain after dropna
            if verbose:
                print(f'Channel {channel} on device {device} localID {localID} has no values after alignment; skipping.')
            continue

        if channel not in ctda_channels:                                                                        # Skip if channel not in ctda_channels
            if verbose:
                print(f'Channel {channel} not found in channel_type_da; skipping.')
            continue

        patch_cols[channel]= vals.rename(channel)                                                               # Store aligned column for this channel

    if not patch_cols:
        if verbose:
            print("No valid data found to update channel_type_da; returning original.")
        return ctda
    
    #===== Construct Patch DataFrame =====
    patch_df= pd.concat(patch_cols.values(), axis=1)                                                            # Combine all columns into a DataFrame indexed by Time
    patch_df= patch_df.reindex(index= ctda_time, 
                               columns= [c for c in ctda_channels if c in patch_df.columns])                    # Reindex to ctda_time and ctda_channels, keeps only channels present in patch_df
    
    if verbose:
        print(f'Constructed patch DataFrame with shape: {patch_df.shape} for updating channel_type_da.')
        display(patch_df)
    #===== Update DataArray =====

    #===== Update DataArray (only write where values exist) =====
    for ch in patch_df.columns:
        col = patch_df[ch].dropna()
        if not col.empty:
            ctda.loc[dict(Time=col.index, channel=ch)] = col.astype(ctda.dtype).values

    if verbose:
        print(f'Updated channel_type_da with data from {len(patch_cols)} channels.')
        print(f'New shape: {ctda.shape}')

    return ctda

################################################################################




################################################################################################################################################################
# Wrappers for renamed, moved, or deprecated functions
################################################################################################################################################################

################################################################################
# Construct Channel Type DataArray
################################################################################

def construct_channel_type_da(values: list | np.ndarray | None = None,
                                test_help_values: bool= False,
                                channels: list= None,
                                channels_ref_dict: dict= None,
                                times: list | np.ndarray= None,
                                name: str= 'channel_type_da',
                                verbose: bool= True,
                                dict_of: str= 'tuples'
                                ) -> xr.DataArray:
    '''
    Construct a Placeholder DataArray for a set of given channels for all timepoints across all devices for a specific register type. Uses construct_base_da internally.
    -----------------------------------
    Parameters:
        values (list or np.ndarray): Values to populate the DataArray. If None, a placeholder array of NaNs will be created.
        test_help_values (bool): If True, use test helper values instead of the main values.
        channels (list): List of channel names to include in the DataArray.
        channels_ref_dict (dict): Dictionary mapping channel names to their reference information.
        times (list): List of timepoints to include in the DataArray. This should be obtained using get_df_dict_timestamps with return_type='list'.
        name (str): Name of the DataArray.
    Returns:
        A DataArray populated with the specified values, or a placeholder array if no values are provided.
    '''
    if verbose: print(f'Warning: construct_channel_type_da is deprecated. Use construct_base_da instead with appropriate parameters.')
    return construct_base_da(
        values= values if values is not None else None,
        unified_dim= channels,
        unified_coord_name= 'channel',
        virtual_coord_names= ['device', 'localID'],
        virtual_map_dict= channels_ref_dict,
        dict_of= dict_of,
        times= times,
        name= name,
        test_help_values= test_help_values,
        verbose= verbose
    )
##--------------------------------------------------------------------------------   Original
# def construct_channel_type_da(values= None, test_help_values= False, channels=None, channels_ref_dict= None, times= None, name='channel_type_da'):
#     '''
#     Construct a Placeholder DataArray for a set of given channels for all timepoints across all devices for a specific register type.
#     -----------------------------------

#     Parameters:
#         values (list or np.ndarray): Values to populate the DataArray. If None, a placeholder array of NaNs will be created.
#         test_help_values (bool): If True, use test helper values instead of the main values.
#         channels (list): List of channel names to include in the DataArray.
#         channels_ref_dict (dict): Dictionary mapping channel names to their reference information.
#         times (list): List of timepoints to include in the DataArray. This should be obtained using get_df_dict_timestamps with return_type='list'.
#         name (str): Name of the DataArray.

#     Returns:
#         A DataArray populated with the specified values, or a placeholder array if no values are provided.
#     '''

#     assert times is not None and channels is not None and channels_ref_dict is not None, "times, channels, and channels_ref_dict must be provided."

#     if values is None and test_help_values is False:
#         print(f'No values provided, and test_help_values is False. Creating zeros placeholder DataArray with {len(times)} timestamps and {len(channels)} channels.')
#         values= np.zeros((len(times), len(channels)),dtype= bool
#                          )
    
#     if values is None and test_help_values is True:
#         print(f'No values provided, but test_help_values is True. Creating test helper DataArray with {len(times)} timestamps and {len(channels)} channels.')
#         value_list= []
#         for time in times:
#             time_values= []
#             for channel in channels:
#                 device, localID = channels_ref_dict[channel]
#                 time_values.append(f'({channel}, {device}, {localID})')
#             value_list.append(time_values)
#         values= np.array(value_list)
#         print(f'Constructed test helper values array with shape: {values.shape}')
#     # if values is not None and test_help_values is True:
#     #     raise ValueError('Cannot provide both values and set test_help_values to True. Choose one or the other.')

#     devices= [channels_ref_dict[channel][0] for channel in channels]
#     localIDs= [channels_ref_dict[channel][1] for channel in channels]

#     da= xr.DataArray(
#         data=values,
#         dims= ['Time', 'channel'],
#         coords= {
#             'Time': times,
#             'channel': channels,
#             'device': ('channel', devices),
#             'localID': ('channel', localIDs)
#         },
#         name= name,

#         attrs= {
#             'description': f'DataArray for channels of type {name}',
#             'source': 'Constructed using construct_channel_type_da function'
#         }
#     )
#     return da
################################################################################




################################################################################
# Construct Channel Lookup DataArray
################################################################################
def construct_channel_lookup_da(channels= None,
                                channels_ref_dict= None,
                                dict_of= 'tuples',
                                verbose= False
):
    '''
    Construct a Channel Lookup DataArray for a set of given channels.
    -----------------------------------

    Parameters:
        channels (list): List of channel names to include in the DataArray.
        channels_ref_dict (dict): Dictionary mapping channel names to their reference information.
        dict_of (str): Specifies the format of channels_ref_dict. Options are 'dicts' or 'tuples'.
        verbose (bool): If True, prints additional information.

    Returns:
        lookup_da (xr.DataArray): A DataArray representing the channel lookup matrix.
        lookup_devices (list): List of unique devices found in the channels_ref_dict.
        lookup_localIDs (list): List of unique local IDs found in the channels_ref_dict.
    '''

    if verbose: print(f'Warning: construct_channel_lookup_da is deprecated. Use construct_lookup_array instead with appropriate parameters.')
    lookup_da, lookup_virtual_coords= construct_lookup_array(
        unified_dim= channels,
        unified_coord_name= 'channel',
        virtual_coord_names= ['device', 'localID'],
        virtual_map_dict= channels_ref_dict,
        dict_of= dict_of,
        name= 'channel_lookup_matrix',
        verbose= verbose
    )
    return lookup_da, lookup_virtual_coords['device'], lookup_virtual_coords['localID']

##--------------------------------------------------------------------------------   Original
# def construct_channel_lookup_da(channels= None, channels_ref_dict= None):
#     '''
#     Construct a Channel Lookup DataArray for a set of given channels.
#     -----------------------------------

#     Parameters:
#         channels (list): List of channel names to include in the DataArray.
#         channels_ref_dict (dict): Dictionary mapping channel names to their reference information.

#     Returns:
#         lookup_da (xr.DataArray): A DataArray representing the channel lookup matrix.
#         lookup_devices (list): List of unique devices found in the channels_ref_dict.
#         lookup_localIDs (list): List of unique local IDs found in the channels_ref_dict.
#     '''

#     lookup_devices= []
#     lookup_localIDs= []

#     for d, l in channels_ref_dict.values():
#         if d not in lookup_devices:
#             lookup_devices.append(d)
#         if l not in lookup_localIDs:
#             lookup_localIDs.append(l)

#     lookup_arr= np.zeros((len(channels), len(lookup_devices), len(lookup_localIDs)), dtype= bool)

#     for i, channel in enumerate(channels):
#         look_device, look_localID = channels_ref_dict[channel]
        
#         device_index= lookup_devices.index(look_device)
#         localID_index= lookup_localIDs.index(look_localID)

#         lookup_arr[i, device_index, localID_index] = True

#     lookup_da= xr.DataArray(
#         data= lookup_arr,
#         dims= ['channel', 'device', 'localID'],
#         coords= {
#             'channel': channels,
#             'device': lookup_devices,
#             'localID': lookup_localIDs
#         },
#         name= 'Channel_lookup_matrix',
#         attrs= {
#             'description': 'Channel lookup matrix indicating presence of channels across devices and local IDs',
#             'source': 'Constructed using construct_channel_lookup_da function'
#         }
#     )
#     return lookup_da, lookup_devices, lookup_localIDs
################################################################################




################################################################################
# Select Channels from CDL
################################################################################
def select_channels_from_C_D_L(lookup_da: xr.DataArray= None,
                               unified_coord_name: str= 'channel',
                               verbose: bool= True,
                               channel: str= None,
                               device: str= None,
                               localID: str= None) -> list:
    '''
    Select channels from a Channel-Device-LocalID lookup DataArray based on specified criteria. Deprecated: use ulookup instead.
    -----------------------------------
    Parameters:
        lookup_da (xr.DataArray): The Channel-Device-LocalID lookup DataArray.
        unified_coord_name (str): Name of the unified/global coordinate dimension (default is 'channel').
        verbose (bool): If True, prints additional information.
        channel (str): The channel name to filter by (optional).
        device (str): The device name to filter by (optional).
        localID (str): The local ID to filter by (optional).
    '''
    if verbose: print(f'Warning: select_channels_from_C_D_L is deprecated. Use ulookup instead with appropriate parameters.')
    return ulookup(
        lookup_arr= lookup_da,
        unified_coord_name= unified_coord_name,
        verbose= verbose,
        channel= channel,
        device= device,
        localID= localID
    )

##--------------------------------------------------------------------------------   Original
# def select_channels_from_C_D_L(lookup_da= None, channel= None, device= None, localID= None):
#     '''
#     Select channels from a Channel-Device-LocalID lookup DataArray based on specified criteria.
#     -----------------------------------

#     Parameters:
#         lookup_da (xr.DataArray): The Channel-Device-LocalID lookup DataArray.
#         channel (str): The channel name to filter by (optional).
#         device (str): The device name to filter by (optional).
#         localID (str): The local ID to filter by (optional).

#     Returns:
#         list: A list of channel names that match the specified criteria.
#     '''

#     sel= {}

#     if channel is not None: sel['channel']= channel
#     if device is not None: sel['device']= device
#     if localID is not None: sel['localID']= localID

#     # Subset
#     subset= lookup_da.sel(**sel) if sel else lookup_da
#     print(f'Subset shape after selection: {subset.shape}')

#     # Reduce across non-channel dimensions
#     other_dims= [d for d in subset.dims if d != 'channel']
#     if other_dims:
#         per_channel_true= subset.any(dim= other_dims)
#     else:
#         per_channel_true= subset.astype(bool)

#     print(f'Per-channel true shape after reduction: {per_channel_true.shape}')

#     return per_channel_true['channel'].values[per_channel_true.values].tolist()
################################################################################




################################################################################
# Test DataArray Indexing
################################################################################
def test_da_indexing(channel_type_da=None, channel_lookup_da=None, channel_list=None, 
                    channel_ref_dict=None, verbose=False, only_errors=False):
    """
    Test slicing and indexing operations for channel data arrays.
    
    Parameters:
    -----------
    channel_type_da : xarray.DataArray
        Data array containing channel type information
        Data array containing channel type information
    channel_lookup_da : xarray.DataArray
        Lookup data array for channels
    channel_list : list
        List of channel names
    channel_ref_dict : dict
        Dictionary with channel reference information
    verbose : bool, default=True
        If True, prints detailed output for each test
    only_errors : bool, default=False
        If True, only reports when errors occur
    
    Returns:
    --------
        dict : Dictionary with test results
    """
    # Initialize the lookup array if needed
    if channel_lookup_da is None:
        lookup_da, devices, local_ids = construct_channel_lookup_da(channels=channel_list, 
                                                                   channels_ref_dict=channel_ref_dict)
    else:
        lookup_da = channel_lookup_da
        # Extract devices and local_ids from lookup_da
        channels = lookup_da.coords['channel'].values.tolist() if channel_list is None else channel_list
        devices = lookup_da.coords['device'].values.tolist() if 'device' in lookup_da.coords else []
        local_ids = lookup_da.coords['localID'].values.tolist() if 'localID' in lookup_da.coords else []
    
    localIDs = local_ids
    test_results = {}
    
    # Helper function to run test case and handle output
    def run_test_case(test_id, description, selection_params):
        try:
            selected_items = select_channels_from_C_D_L(lookup_da, **selection_params)
            test_results[test_id] = {"success": True, "selected_items": selected_items}
            
            if verbose and not only_errors:
                print(f"\n{test_id}) {description}:")
                print(selected_items)
                if channel_type_da is not None:
                    display(channel_type_da.sel(channel=selected_items).to_dataframe())
                    print("\nWithout device and localID columns:")
                    display(channel_type_da.sel(channel=selected_items).to_dataframe().unstack('channel')
                           .drop(columns=['device', 'localID']))
            return True
            
        except Exception as e:
            test_results[test_id] = {"success": False, "error": str(e)}
            if verbose or only_errors:
                print(f"\n{test_id}) {description} - ERROR: {str(e)}")
            return False
    
    # Define all test cases
    test_cases = [
        ("A", "Slice devices by label range, and take the 2nd localID ('DOPort1')", 
         {"device": slice('Behavior1','Behavior5'), "localID": 'DOPort1'}),
        ("B", "Slice localIDs by label range; any device", 
         {"localID": slice('DOPort0','DOPort2')}),
        ("C", "List of devices; list of localIDs", 
         {"device": ['Behavior0','Behavior2'], "localID": ['DOPort0','DOPort1']}),
        ("D", "Slice channels by label range", 
         {"channel": slice('Nosepoke2','Nosepoke6')}),
        ("E", "Exact single cell", 
         {"channel": 'Nosepoke4', "device": 'Behavior1', "localID": 'DOPort1'}),
        ("F", "All channels (wildcard)", 
         {"channel": None, "device": None, "localID": None}),
        ("G", "Slice using numeric index for channels", 
         {"channel": channels[0:2] if len(channels) >= 2 else []}),
        ("H", "Slice using numeric index for devices", 
         {"device": devices[0:2] if len(devices) >= 2 else []}),
        ("I", "Slice using numeric index for localIDs", 
         {"localID": localIDs[0:2] if len(localIDs) >= 2 else []}),
        ("J", "Slice using label range for devices and localIDs", 
         {"device": slice('Behavior1', 'Behavior2'), "localID": slice('DOPort0', 'DOPort2')}),
        ("K", "Slice using numeric index for devices and localIDs", 
         {"device": devices[1:3] if len(devices) >= 3 else [], 
          "localID": localIDs[0:3] if len(localIDs) >= 3 else []}),
        ("L", "Slice using label range for channels", 
         {"channel": slice('Nosepoke2', 'Nosepoke6'), "device": None, "localID": None}),
        ("M", "Slice using numeric index for channels", 
         {"channel": channels[2:7] if len(channels) >= 7 else [], 
          "device": devices[:], "localID": localIDs[:]})
    ]

  

    
    # Run all test cases
    all_success = True
    for test_id, description, params in test_cases:
        success = run_test_case(test_id, description, params)
        all_success = all_success and success
    
    # Report summary
    if verbose:
        success_count = sum(1 for result in test_results.values() if result["success"])
        total_count = len(test_results)
        
        if only_errors:
            if all_success:
                print("\nAll tests passed successfully!")
            else:
                failures = [(test_id, result) for test_id, result in test_results.items() if not result["success"]]
                print(f"\n{len(failures)}/{total_count} tests failed.")
        else:
            print(f"\nTest results: {success_count}/{total_count} tests passed.")
    
    return test_results

################################################################################



################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original

################################################################################





################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original

################################################################################
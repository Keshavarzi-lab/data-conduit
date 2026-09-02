"""
Virtual Arrays.

Virtual arrays allow users to define n-dimensional lookup matrices that map
from a series of input coordinates to a given output coordinate.

    - Input coordinates are referred to as virtual coordinates, and with
      output coordinates being referred to as global or unified coordinates.
    - These virtual arrays are used to perform flexible and efficient lookup
      query operations, as well as support for constructing custom xarray
      accessors.
"""


# check 

################################################################################
# Imports
################################################################################
from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr

################################################################################






################################################################################
# Private Helper Functions
################################################################################

#===============================================================================
# 1| Generate Test Values
#===============================================================================
def _generate_test_values(
        times: list | np.ndarray,
        global_coords: list,
        virtual_map: dict,
        virtual_coord_names: list,
        dict_of: str = 'dicts',
        verbose: bool = False,
) -> np.ndarray:
    '''
    Generate human-readable test values for debugging DataArray construction.  
    Creates string values in following format: "((global_coord, vcoord1: val1, vcoord2: val2, ...))".
    
    -------------------------------
    Parameters:
        times: list | np.ndarray
            A list or array of timepoints.
        global_coords: list
            A list of global coordinates.
        virtual_map: dict
            A dictionary mapping global coordinates to virtual coordinates.
        virtual_coord_names: list
            A list of names for the virtual coordinate dimensions.
        dict_of: str
            Specifies the format of virtual_map_dict. Options are 'dicts' or 'tuples'.
        verbose: bool
            If True, function will print additional information during execution. Default is False.
    Returns:
        np.ndarray: 
            2D array of test value strings with shape (len(times), len(global_coords))
    '''

    if verbose:
        print(f'No values provided, test_values is True. Creating test values DataArray with {len(times)} timestamps and {len(global_coords)} unified coordinates.')    # noqa

    value_list= []
    
    for _ in times:
        time_values = []
        for unified in global_coords:
            virtual_coords = virtual_map[unified]

            if dict_of == 'dicts':
                if not isinstance(virtual_coords, Mapping): 
                    raise TypeError(f'Expected virtual_map to be dict-of-dicts when dict_of="dicts", but got {type(virtual_coords)} for coordinate {unified}')
                
                # Format virtual coordinates as comma-separated "key: value" pairs
                # Used for creating human-readable test values
                # Example output: "row: 5, col: 3, layer: 2"
                
                coord_str = ', '.join([f'{k}: {v}' for k, v in virtual_coords.items()])

            elif dict_of == 'tuples':
                
                if not isinstance(virtual_coords, (tuple, list)):                           # noqa
                    raise TypeError(f'Expected virtual_map to be dict-of-tuples/lists when dict_of="tuples", but got {type(virtual_coords)} for unified coordinate {unified}')
                
                if len(virtual_coords) != len(virtual_coord_names): 
                    raise ValueError(f"Tuple length mismatch for {unified}: len(entry)={len(virtual_coords)} vs len(virtual_coord_names)={len(virtual_coord_names)}")
                
                # Map coord names to tuple values: "name1: val1, name2: val2, ..."
                coord_str = ", ".join([f"{k}: {v}" for k, v in zip(virtual_coord_names, virtual_coords, strict=True)])

            else:                   
                raise TypeError(f"dict_of parameter must be either 'dicts' or 'tuples', got: {dict_of}")

            time_values.append(f'({unified}, {coord_str})')

        value_list.append(time_values)
    values = np.array(value_list)
    if verbose: 
        print(f'Constructed test helper values array with shape: {values.shape}')
    return values


#===============================================================================      

################################################################################






################################################################################
# Construct Base DataArray
################################################################################


def construct_data_array(virtual_map: dict= None,
                         global_coord_name: str= 'global_coord',
                         virtual_coord_names: list= None,
                         times: list | np.ndarray= None,
                         dict_of: str= 'dicts',
                         values: list | np.ndarray | None= None,
                         name: str= 'data_array',
                         test_values: bool= False,
                         verbose: bool= False,
                         da_attrs: dict | None= None
                         ) -> xr.DataArray:
    '''
    Construct a base DataArray for a given set of global coordinates for all timepoints and their associated virtual coordinates.
    
    virtual_map formats:
        dict_of="dicts":  {global: {vcoord_name: vcoord_value, ...}, ...}
        dict_of="tuples": {global: (v0, v1, ...), ...} aligned with virtual_coord_names
    -------------------------------

    Parameters:
        virtual_map: dict
            A dictionary mapping global coordinates to virtual coordinates.
        global_coord_name: str
            The name of the global coordinate dimension.
        virtual_coord_names: list
            A list of names for the virtual coordinate dimensions.
        times: list | np.ndarray
            A list or array of timepoints to include in the DataArray. Obtained using collect_timestamps_dict() or collect_timestamps_nested_dict() with return_type= 'list'.
        dict_of: str
            Specifies the format of virtual_map_dict. Options are 'dicts' or 'tuples'.
        values: list | np.ndarray | None
            Values to populate the DataArray. If None, a placeholder array of NaNs will be created.
        name: str
            The name of the DataArray.
        test_values: bool
            If True, function will use test values instead of the provided values. Default is False. Useful for debugging.
        verbose: bool
            If True, function will print additional information during execution. Default is False.
        da_attrs: dict | None
            A dictionary of attributes to assign to the DataArray. Default is None.

    Returns:
        xr.DataArray
            A DataArray with dimensions for time, global coordinates, and virtual coordinates.
    '''

    dict_of= dict_of.lower()

    if dict_of not in ("dicts", "tuples"):  
        raise ValueError(f"dict_of must be either 'dicts' or 'tuples', got: {dict_of}")
    if times is None:                       
        raise ValueError("times parameter must be provided and cannot be None.")
    if global_coord_name is None:           
        raise ValueError("global_coord_name parameter must be provided and cannot be None.")
    if virtual_coord_names is None:         
        raise ValueError("virtual_coord_names parameter must be provided and cannot be None.")
    if virtual_map is None:                 
        raise ValueError("virtual_map parameter must be provided and cannot be None.")


    if da_attrs is None:
        if verbose: 
            print("No da_attrs provided, using defaults.")
        da_attrs= {'description': f'Base DataArray for unified coordinates of type {name}',
                       'source': 'Constructed using construct_base_da function'}
    elif not isinstance(da_attrs, dict):
        if verbose: 
            print("da_attrs is not a dict, using defaults.")
        da_attrs= {'description': f'Base DataArray for unified coordinates of type {name}',
                            'source': 'Constructed using construct_base_da function'}
    else:
        if verbose: 
            print("Using provided da_attrs dictionary.")


    #==== Validate Virtual Map Format
    if dict_of == "tuples" and any(isinstance(v, dict) for v in virtual_map.values()):
                                            raise TypeError("dict_of='tuples' but at least one entry is a dict. Use dict_of='dicts'.")

    if dict_of == "dicts" and any(isinstance(v, (tuple, list)) for v in virtual_map.values()):                                              # noqa
                                            raise TypeError("dict_of='dicts' but at least one entry is a tuple/list. Use dict_of='tuples'.")

    

    #==== Create data to populate DataArray 
    global_coords= list(virtual_map.keys())

    if values is not None and test_values is True:    
        print('Warning: Both values provided and test_values is True. Using provided values.')

    if values is None and test_values is False:   
        if verbose: 
            print(f'No values provided, test_values is False. Creating NaNs placeholder DataArray with {len(times)} timestamps and {len(global_coords)} unified coordinates.') # noqa
        values = np.full((len(times), len(global_coords)), np.nan, dtype=float)
        

    #==== Create test values if specified
    if values is None and test_values is True:
            values= _generate_test_values(times= times,
                                          global_coords= global_coords,
                                          virtual_map= virtual_map,
                                          virtual_coord_names= virtual_coord_names,
                                          dict_of= dict_of,
                                          verbose=verbose
                                          )

    values = np.asarray(values)
    expected = (len(times), len(global_coords))
    if values.shape != expected:
        raise ValueError(f"values has shape {values.shape}, expected {expected}")


    #==== Build Coordinates

    # i) Primary/Base Coordinates
    coords= {'Time': times,
             global_coord_name: global_coords,
    }
    

    # ii) Virtual Coordinates (dict-of-dicts/ Named Fields)

    if dict_of== 'dicts':
            for vcoord in virtual_coord_names:
                    try: 
                        vcoord_values= [virtual_map[global_coord][vcoord] 
                                        for global_coord in global_coords]
                    
                    except KeyError as e:
                        raise KeyError(f'Missing key {e} in virtual_map entries (dict_of="dicts")') from e
                    

                    # Add virtual coordinate to coords dict
                    coords[vcoord] = (global_coord_name, vcoord_values)


    # iii) Virtual Coordinates (dict-of-tuples/ Positional Fields)

    elif dict_of== 'tuples':
            for idx, vcoord in enumerate(virtual_coord_names):
                    try: 
                        vcoord_values = [virtual_map[global_coord][idx] 
                                        for global_coord in global_coords]
                    except IndexError as e:
                        raise IndexError(f'Index {idx} out of range for virtual_map entries (dict_of="tuples")') from e
                    
                    # Add virtual coordinate to coords dict
                    coords[vcoord] = (global_coord_name, vcoord_values)


    else: 
        raise TypeError(f"dict_of parameter must be either 'dicts' or 'tuples', got: {dict_of}")

    #=====================================
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
    #=====================================

    #==== Create DataArray
    data_array= xr.DataArray(data= values,
                             dims= ['Time', global_coord_name],
                             coords= coords,
                             name= name,
                             attrs= da_attrs
                             )
    if verbose:
        print(f'Constructed DataArray "{name}" with dimensions: {data_array.dims} and shape: {data_array.shape}')
    
    return data_array

################################################################################




################################################################################
# Construct Lookup Array
################################################################################

def construct_lookup_array(virtual_map: dict= None,
                           global_coord_name: str= 'global_coord',
                           virtual_coord_names: list= None,
                           dict_of: str= 'dicts',
                            name: str= 'lookup_matrix',
                            verbose: bool= False,
                            lookup_attrs: dict | None= None
                            ) -> tuple[xr.DataArray, dict[str, list]]:
    '''
    Construct a lookup DataArray for a given set of global coordiantes and their associated virtual coordinates.
    
    virtual_map formats:
        dict_of="dicts":  {global: {vcoord_name: vcoord_value, ...}, ...}
        dict_of="tuples": {global: (v0, v1, ...), ...} aligned with virtual_coord_names
    -------------------------------
    Parameters:
        virtual_map: dict
            A dictionary mapping global coordinates to virtual coordinates.
        global_coord_name: str
            The name of the global coordinate dimension.
        virtual_coord_names: list
            A list of names for the virtual coordinate dimensions.
        dict_of: str
            Specifies the format of virtual_map_dict. Options are 'dicts' or 'tuples'.
        name: str
            The name of the DataArray.
        verbose: bool
            If True, function will print additional information during execution. Default is False.
        lookup_attrs: dict | None
            A dictionary of attributes to assign to the DataArray. Default is None.

    Returns:
        lookup_da (xr.DataArray): 
            A DataArray representing the lookup matrix.
        lookup_virtual_coords (dict):
            A dictionary containing lists of unique virtual coordinate values for each virtual coordinate dimension. 
            Keys are virtual coordinate names, values are lists of unique values. 
    '''

    #==== Validate Inputs
    dict_of= dict_of.lower()
    
    if dict_of not in ("dicts", "tuples"):
        raise ValueError(f"dict_of must be either 'dicts' or 'tuples', got: {dict_of}")

    if virtual_map is None:
        raise ValueError("virtual_map parameter must be provided and cannot be None.")
    if global_coord_name is None:
        raise ValueError("global_coord_name parameter must be provided and cannot be None.")
    if virtual_coord_names is None:
        raise ValueError("virtual_coord_names parameter must be provided and cannot be None.")
    
    if not isinstance(virtual_coord_names, list) or len(virtual_coord_names) == 0:
        raise ValueError("virtual_coord_names must be a non-empty list.")
    
    if dict_of == "tuples" and any(isinstance(v, dict) for v in virtual_map.values()):
        raise TypeError("dict_of='tuples' but at least one entry is a dict. Use dict_of='dicts'.")

    if dict_of == "dicts" and any(isinstance(v, (tuple, list)) for v in virtual_map.values()):              # noqa
        raise TypeError("dict_of='dicts' but at least one entry is a tuple/list. Use dict_of='tuples'.")
    

    #=== Default Attributes
    if lookup_attrs is None:
        if verbose: 
            print("No lookup_attrs provided, using defaults.")
        lookup_attrs=  {'description': 'Lookup matrix indicating presence of unified coordinates across virtual coordinate dimensions',
                        'source': 'Constructed using construct_lookup_array function'
        }
    elif not isinstance(lookup_attrs, dict):
        raise TypeError(f"lookup_attrs must be a dict or None, got: {type(lookup_attrs)}")

    #==== Get Global Coordinates
    global_coords = list(virtual_map.keys())
    if len(global_coords) == 0:
        raise ValueError("virtual_map is empty: no global coordinates to build a lookup from.")
    
    if verbose:
        print(f'Constructing 1D lookup for {len(global_coords)} {global_coord_name}s')
        print(f'Virtual coordinates: {virtual_coord_names}')

    #==== Bild Coordinate Arrrays for Each Virtual Coord
    coords = {global_coord_name: global_coords}
    lookup_virtual_coords = {}

    for vcoord in virtual_coord_names:
        
        values= []  # Extracts values for specified virtual coordinate

        for gcoord in global_coords:
            vcoord_entry = virtual_map[gcoord]

            if dict_of == 'tuples':
                if not isinstance(vcoord_entry, (tuple, list)):                                                 # noqa
                    raise TypeError(f'Expected virtual_map to be dict-of-tuples/lists when dict_of="tuples",'
                                            f'but got {type(vcoord_entry)} for global coordinate {gcoord}')
                if len(vcoord_entry) != len(virtual_coord_names):
                    raise ValueError(f'Tuple length mismatch for {gcoord}: len(entry)={len(vcoord_entry)} vs len(virtual_coord_names)={len(virtual_coord_names)}')
                
                # Get value by position
                idx= virtual_coord_names.index(vcoord)
                values.append(vcoord_entry[idx])
               
            elif dict_of == 'dicts':
                if not isinstance(vcoord_entry, Mapping):
                    raise TypeError(f'Expected virtual_map to be dict-of-dicts when dict_of="dicts",'
                                            f'but got {type(vcoord_entry)} for global coordinate {gcoord}')
                
                if vcoord not in vcoord_entry:
                    raise KeyError(f'Missing key {vcoord} in virtual_map entries (dict_of="dicts") for global coordinate {gcoord}')
                # Get value by key
                values.append(vcoord_entry[vcoord])
            else:
                raise TypeError(f"dict_of parameter must be either 'dicts' or 'tuples', got: {dict_of}")
            
        # Store as coordinate indexed by global coordinate
        coords[vcoord] = (global_coord_name, values)

        # Store unique virtual coordinate values for lookup_virtual_coords
        unique_vals = list(dict.fromkeys(values)) 
        lookup_virtual_coords[vcoord] = unique_vals
        
        if verbose:    
            print(f'  {vcoord}: {len(unique_vals)} unique values')

    #==== Create 1D Lookup DataArray
    lookup_array= xr.DataArray(data = np.ones(len(global_coords), dtype=bool),
                                 dims= [global_coord_name],
                                    coords= coords,
                                    name= name,
                                    attrs= lookup_attrs
                                    )
    if verbose:
        print(f'Constructed lookup DataArray "{name}" with dimensions: {lookup_array.dims} and shape: {lookup_array.shape}')
    return lookup_array, lookup_virtual_coords

################################################################################






################################################################################
# Update DataArray 
################################################################################

def update_data_array(data_array: xr.DataArray = None,
                      virtual_map: dict = None,
                      dict_of: str = 'dicts',
                      dfs_dict: dict = None,
                      fill_value: any = np.nan,
                      verbose: bool = False
                    ) -> xr.DataArray:
    '''
    Update DataArray with values from dfs_dict based on mapping provided in virtual_map.
    
    Format of dfs_dict and virtual_map should match, i.e. if dfs_dict is indexable as 
    [global]: [virtual_1][virtual_2].
    --------------------------------
    
    Parameters:
        data_array: xr.DataArray
            The DataArray to be updated. Should have dimensions for time and global coordinates, and virtual coordinate information in its coordinates.
        virtual_map: dict
            A dictionary mapping global coordinates to virtual coordinates. Format should match the structure of dfs_dict 
            for lookups. Can be in dict-of-dicts or dict-of-tuples format.
        dict_of: str
            Specifies the format of virtual_map and dfs_dict. Options are 'dicts' or 'tuples'. Default is 'dicts'.
        dfs_dict: dict
            A nested dictionary containing the source data for updating the DataArray. Should be structured to allow 
            lookups based on the virtual coordinates specified in virtual_map. For example, if virtual_map is 
            dict-of-dicts with virtual coordinates 'device' and 'localID', dfs_dict should be structured as 
            dfs_dict[device][localID][global_coord] to retrieve the value for a given global coordinate.
        fill_value: any
            A value to use for filling entries in the DataArray when lookups fail or when virtual coordinate information is missing. Default is np.nan.
        verbose: bool
            If True, function will print additional information during execution. Default is False.
    
    Returns:
        xr.DataArray
            The updated DataArray with values filled in from dfs_dict based on the mapping in virtual_map
    '''

    #===== Validate Inputs
    if data_array is None:                          
        raise ValueError("data_array parameter must be provided and cannot be None.")
    if virtual_map is None:                         
        raise ValueError("virtual_map parameter must be provided and cannot be None.")
    if dict_of.lower() not in ('dicts', 'tuples'):  
        raise ValueError(f"dict_of must be either 'dicts' or 'tuples', got: {dict_of}")
    if dfs_dict is None:                            
        raise ValueError("dfs_dict parameter must be provided and cannot be None.")

    if verbose:
        print(f'Updating DataArray with shape {data_array.shape}')
        print(f'Virtual map format: {dict_of}')
        print(f'Number of global coordinates: {len(virtual_map)}')


    #===== Collect Dimensions 

    time_dim = [d for d in data_array.dims if d.lower() == 'time'][0]   # Identify time dimension
    global_dim = [d for d in data_array.dims if d.lower() != 'time'][0] # Identify global coordinate dimension

    da_time = data_array.get_index(time_dim)                            # Get time coordinate values
    
    # If fill_value provided, pre-fill entire array
    if fill_value is not None:
        data_array[:] = fill_value
    #===== Prebuild source columns dict to avoid redundant lookups in dfs_dict
    source_cols_dict = {}
    
    for global_coord, virtual_vals in virtual_map.items():
        try: 
            current = dfs_dict

            if dict_of == 'tuples':
                for key in virtual_vals:
                    current = current[key]
            else:  # dicts
                for key in virtual_vals.values():
                    current = current[key]
            
            # Store the source column in the dict
            source_cols_dict[global_coord] = current[global_coord] if isinstance(current, pd.DataFrame) else current
        except (KeyError, TypeError) as e:
            if verbose:
                print(f"Warning: Could not retrieve source column for global coordinate '{global_coord}' with virtual values {virtual_vals}. Error: {e}")
            source_cols_dict[global_coord] = pd.Series(fill_value, index=da_time)  # Fill with default value if retrieval fails
            

    # Catch case where no valid data to update
    if not source_cols_dict:
        return data_array

    #===== Bulk Update All DataArray Columns at once

    combined_df = pd.DataFrame(source_cols_dict)
    # Align combined_df with data_array time coordinate
    combined_df = combined_df.reindex(da_time, fill_value=fill_value if fill_value is not None else np.nan)

    # Single assignment to update entire DataArray
    valid_mask = combined_df.notna()
    
    for col in combined_df.columns:
        if verbose:
            print(f"Updating global coordinate '{col}' with {valid_mask[col].sum()} valid entries out of {len(valid_mask[col])} total.")
        mask = valid_mask[col]
        if mask.any():
            data_array.loc[{time_dim: da_time[mask], global_dim: col}] = combined_df.loc[mask, col].values
    return data_array

################################################################################


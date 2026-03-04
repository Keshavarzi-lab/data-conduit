################################################################################
# Imports
################################################################################


import pandas as pd
import numpy as np
import xarray as xr
from collections.abc import Mapping

################################################################################







################################################################################
# Private Helper Functions
################################################################################

#===============================================================================
# 1| Generate Test Values
#===============================================================================

def _generate_test_values(times: list | np.ndarray,
                          global_coords: list,
                          virtual_map: dict,
                          virtual_coord_names: list,
                          dict_of: str= 'dicts',
                          verbose: bool= False
                          ) -> np.ndarray:

    '''
    Generate human-readable test values for debugging DataArray construction.
    
    Creates string values in format: "(global_coord, vcoord1: val1, vcoord2: val2, ...)"
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
        print(f'No values provided, test_values is True. Creating test values DataArray with {len(times)} timestamps and {len(global_coords)} unified coordinates.')
    
    value_list= []
    
    for _ in times:
        time_values = []
        for unified in global_coords:
            virtual_coords = virtual_map[unified]

            if dict_of == 'dicts':
                if not isinstance(virtual_coords, Mapping): 
                                    raise TypeError(f'Expected virtual_map to be dict-of-dicts when dict_of="dicts", but got {type(virtual_coords)} for unified coordinate {unified}')
                
                # Format virtual coordinates as comma-separated "key: value" pairs
                # Used for creating human-readable test values
                # Example output: "row: 5, col: 3, layer: 2"
                
                coord_str = ', '.join([f'{k}: {v}' for k, v in virtual_coords.items()])

            elif dict_of == 'tuples':
                
                if not isinstance(virtual_coords, (tuple, list)): 
                                    raise TypeError(f'Expected virtual_map to be dict-of-tuples/lists when dict_of="tuples", but got {type(virtual_coords)} for unified coordinate {unified}')
                
                if len(virtual_coords) != len(virtual_coord_names): 
                                    raise ValueError(f"Tuple length mismatch for {unified}: len(entry)={len(virtual_coords)} vs len(virtual_coord_names)={len(virtual_coord_names)}")
                
                # Map coord names to tuple values: "name1: val1, name2: val2, ..."
                coord_str = ", ".join([f"{k}: {v}" for k, v in zip(virtual_coord_names, virtual_coords)])

            else:                   raise TypeError(f"dict_of parameter must be either 'dicts' or 'tuples', got: {dict_of}")

            time_values.append(f'({unified}, {coord_str})')

        value_list.append(time_values)
    values = np.array(value_list)
    if verbose: print(f'Constructed test helper values array with shape: {values.shape}')
    return values


#===============================================================================




#===============================================================================
# 2| Apply Global Coordinate Selector 
#===============================================================================
def _apply_global_selector(results: xr.DataArray= None,
                           dim_name: str= None,
                           selector: any= None,
                            verbose: bool= False
                           )-> xr.DataArray:
    '''
    Apply selector to global coordinate dimension of lookup DataArray.
    Uses direct indexing via .sel() method.
    -------------------------------
    Parameters:
        results: xr.DataArray
            The current results DataArray to apply selection on.
        dim_name: str
            The name of the global coordinate dimension.
        selector: any
            The selection criteria (single value, list, slice).
        verbose: bool
            If True, function will print additional information during execution. Default is False.
    Returns:
        xr.DataArray:
            The filtered DataArray after applying the global coordinate selector.
    
    Handles:
        Single values: global_coord='coord_5'
        Lists: global_coord=['coord_1', 'coord_3', 'coord_7']
        Slices: global_coord=slice('coord_2', 'coord_6'). Slice endpoints are inclusive.
    '''

    #== Label-Based Slicing 
    if isinstance(selector, slice):
          results= results.sel({dim_name: selector})
          if verbose: print(f'Applied slice selector on {dim_name}: {selector}, resulting shape: {results.shape}')

    #== List-Based Selection
    elif isinstance(selector, (list, tuple)):
        results= results.sel({dim_name: selector})
        if verbose: print(f'Applied list selector on {dim_name}: {selector}, resulting shape: {results.shape}')
    
    #== Single Value Selection
    else:
        results= results.sel({dim_name: [selector]})
        if verbose: print(f'Applied single value selector on {dim_name}: {selector}, resulting shape: {results.shape}')
    return results
    

#===============================================================================




#===============================================================================
# 3| Apply Virtual Coordinate Selector
#===============================================================================

def _apply_virtual_selector(results: xr.DataArray= None,
                            coord_name: str= None,
                            selector: any= None,
                            lookup_array: xr.DataArray= None,
                            verbose: bool= False
                            ) -> xr.DataArray:
    '''
    Apply selector to virtual coordinate dimension of lookup DataArray.
    Filtering using .where() method.
    -------------------------------
    Parameters:
        results: xr.DataArray
            The current results DataArray to apply selection on.
        coord_name: str
            The name of the virtual coordinate dimension.
        selector: any
            The selection criteria (single value, list, slice).
        lookup_array: xr.DataArray
            The original lookup DataArray for reference.
        verbose: bool
            If True, function will print additional information during execution. Default is False.
    Returns:
        xr.DataArray:
            The filtered DataArray after applying the virtual coordinate selector.
    Handles:
        Single values: device='Behavior1'
        Lists: device=['Behavior0', 'Behavior2']
        Slices: device=slice('Behavior1', 'Behavior5'). Label-based, converted to list
    '''

    coord_values= results.coords[coord_name]

    #== Slicing
    if isinstance(selector, slice):
        unique_vals= sorted(set(lookup_array.coords[coord_name].values.tolist()))

        start_idx= 0
        stop_idx= len(unique_vals)

        if selector.start is not None:
            try:
                start_idx = unique_vals.index(selector.start)
            except ValueError as e:
                raise ValueError(f'Slice start value "{selector.start}" not found in virtual coordinate "{coord_name}" values.') from e
        
        if selector.stop is not None:
            try:
                stop_idx = unique_vals.index(selector.stop) + 1  # Inclusive
            except ValueError as e:
                raise ValueError(f'Slice stop value "{selector.stop}" not found in virtual coordinate "{coord_name}" values.') from e
        
        slice_vals= unique_vals[start_idx: stop_idx]
        mask = coord_values.isin(slice_vals)

        if verbose:             
            print(f'Slice [{selector.start}:{selector.stop}] → {len(slice_vals)} allowed values')
        
    #== List-Based Selection
    elif isinstance(selector, (list, tuple)):
        mask = coord_values.isin(selector)
        if verbose:
            print(f'List selector on {coord_name}: {selector}, resulting shape: {results.shape}')
    #== Single Value Selection
    else:
        mask = coord_values == selector
        if verbose:
            print(f'Single value selector on {coord_name}: {selector}, resulting shape: {results.shape}')
    #== Apply Mask
    results= results.where(mask, drop= True)
    if verbose:
        print(f'After applying selector on {coord_name}, resulting shape: {results.shape}')
    return results
#===============================================================================



#===============================================================================
#
#===============================================================================


#===============================================================================




#===============================================================================
#
#===============================================================================


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
            A list or array of timepoints to include in the DataArray. Should be obtained using collect_timestamps_dict() or collect_timestamps_nested_dict() with  return_type= 'list'.
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

    if dict_of not in ("dicts", "tuples"):  raise ValueError(f"dict_of must be either 'dicts' or 'tuples', got: {dict_of}")
    if times is None:                       raise ValueError("times parameter must be provided and cannot be None.")
    if global_coord_name is None:           raise ValueError("global_coord_name parameter must be provided and cannot be None.")
    if virtual_coord_names is None:         raise ValueError("virtual_coord_names parameter must be provided and cannot be None.")
    if virtual_map is None:                 raise ValueError("virtual_map parameter must be provided and cannot be None.")



    if da_attrs is None:
        if verbose: print("No da_attrs provided, using defaults.")
        da_attrs= {'description': f'Base DataArray for unified coordinates of type {name}',
                       'source': 'Constructed using construct_base_da function'}
    elif not isinstance(da_attrs, dict):
        if verbose: print("da_attrs is not a dict, using defaults.")
        da_attrs= {'description': f'Base DataArray for unified coordinates of type {name}',
                            'source': 'Constructed using construct_base_da function'}
    else:
        if verbose: print("Using provided da_attrs dictionary.")



    #==== Validate Virtual Map Format
    if dict_of == "tuples" and any(isinstance(v, dict) for v in virtual_map.values()):
                                            raise TypeError("dict_of='tuples' but at least one entry is a dict. Use dict_of='dicts'.")

    if dict_of == "dicts" and any(isinstance(v, (tuple, list)) for v in virtual_map.values()):
                                            raise TypeError("dict_of='dicts' but at least one entry is a tuple/list. Use dict_of='tuples'.")

    

    #==== Create data to populate DataArray 
    global_coords= list(virtual_map.keys())

    if values is not None and test_values is True:    
        print('Warning: Both values provided and test_values is True. Using provided values.')

    if values is None and test_values is False:   
        if verbose: 
            print(f'No values provided, test_values is False. Creating NaNs placeholder DataArray with {len(times)} timestamps and {len(global_coords)} unified coordinates.')
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
                    try: vcoord_values= [virtual_map[global_coord][vcoord] 
                                         for global_coord in global_coords]
                    
                    except KeyError as e:
                        raise KeyError(f'Missing key {e} in virtual_map entries (dict_of="dicts")') from e
                    

                    # Add virtual coordinate to coords dict
                    coords[vcoord] = (global_coord_name, vcoord_values)


    # iii) Virtual Coordinates (dict-of-tuples/ Positional Fields)

    elif dict_of== 'tuples':
            for idx, vcoord in enumerate(virtual_coord_names):
                    try: vcoord_values = [virtual_map[global_coord][idx] 
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
            A dictionary containing lists of unique virtual coordinate values for each virtual coordinate dimension. Keys are virtual coordinate names, values are lists of unique values. 
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

    if dict_of == "dicts" and any(isinstance(v, (tuple, list)) for v in virtual_map.values()):
        raise TypeError("dict_of='dicts' but at least one entry is a tuple/list. Use dict_of='tuples'.")
    

    #=== Default Attributes
    if lookup_attrs is None:
        if verbose: print("No lookup_attrs provided, using defaults.")
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
                if not isinstance(vcoord_entry, (tuple, list)):
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
        
        if verbose:    print(f'  {vcoord}: {len(unique_vals)} unique values')

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
# Select Global Coordinates from Lookup Array
################################################################################

def ulookup(lookup_array: xr.DataArray= None,
            global_coord_name: str= 'global_coord',
            verbose: bool= False,
            **selectors,
            )-> list:
    '''
    Selects global coordinates from a 1D lookup DataArray based on specified criteria across virtual coordinate dimensions.
    Supports both positional- and label-based indexing and combinations thereof.
    
    Supports:
        Single values: device='Behavior1'
        Lists: device=['Behavior0', 'Behavior2']
        Slices: device=slice('Behavior1', 'Behavior5')
        None/wildcard: device=None (select all)
    -------------------------------

    Parameters:
        lookup_array: xr.DataArray
            A 1D lookup DataArray constructed using construct_lookup_array function.
        global_coord_name: str
            The name of the global coordinate dimension.
        verbose: bool
            If True, function will print additional information during execution. Default is False.
        **selectors:
            Keyword arguments specifying selection criteria for virtual coordinate dimensions. Keys are virtual coordinate names, values can be single values, lists, slices, or None.

    Returns:
        list:
            A list of global coordinates that match the specified selection criteria.
    '''

    if lookup_array is None: 
        raise ValueError("lookup_array parameter must be provided and cannot be None.")
    if global_coord_name not in lookup_array.dims:
        raise ValueError(f'global_coord_name "{global_coord_name}" not found in lookup_array dimensions: {lookup_array.dims}')
    
    if verbose:
        print(f'Selecting from lookup DataArray with dimensions: {lookup_array.dims} and shape: {lookup_array.shape}')
        print(f'Selectors provided: {selectors}')
    
    #==== Build Selection Dict
    
    sel= {}

    # Remove None selectors (wildcards)
    sel= {key: value for key , value in selectors.items() if value is not None}

    if not sel:
        if verbose: print('No selectors provided (all None), returning all global coordinates.')
        return lookup_array[global_coord_name].values.tolist()
    
    if verbose: print(f'Applying following selectors: {sel}')
    
    #==== Apply Selection
    
    # Initialise results with all True mask

    results= lookup_array

    # Apply each selector

    for coord_name , selector_value in sel.items():
           
        #== Direct selection on global coordinate dimension
        if coord_name == global_coord_name:
               results= _apply_global_selector(results= results,
                                               dim_name= global_coord_name,
                                               selector= selector_value,
                                               verbose= verbose
                                               )
               
        #== Selection on virtual coordinate dimensions
        elif coord_name in results.coords:
               results= _apply_virtual_selector(results= results,
                                                coord_name= coord_name,
                                                selector= selector_value,
                                                lookup_array= lookup_array,
                                                verbose= verbose
                                                )
        #== Invalid Coordinate Filter
        else:
            available= [global_coord_name] + [coord for coord in lookup_array.coords.keys() if coord != global_coord_name]
            raise ValueError(f'Coordinate name "{coord_name}" not found in lookup_array coordinates: {available}')
    
    #==== Extract Selected Global Coordinates
    selected = results.coords[global_coord_name].values.tolist()
    if verbose:
        print(f'Selected {len(selected)} {global_coord_name}(s)')

        if len(selected) <= 10:     print(f'  Result: {selected}')
        else:                       print(f'  Result (first 10): {selected[:10]}...')
    
    return selected

################################################################################




################################################################################
# Lookup Accessor with Auto-Construction of Lookup Array
################################################################################

@xr.register_dataarray_accessor('ulookup')
class LookupAccessorConstructor:
    '''
    Base class for constructing custom xarray DataArray accessors for lookup operations with auto-construction of lookup arrays.
    '''

    def __init__(self,
                 data_array: xr.DataArray
                 ):
          
        self._da = data_array
          
        lookup_array = self.lookup_constructor(lookup_array= None,
                                                 global_coord_name= None,
                                                 dict_of= None,
                                                 verbose= False
                                                 )
        self.lookup_array = lookup_array

    def lookup_constructor(self,
                           lookup_array: xr.DataArray= None,
                           global_coord_name: str= 'global_coord',
                           dict_of: str= 'dicts',
                           verbose: bool= False
                           ):
        '''
        Construct or validate the lookup DataArray for the accessor.
        '''

        # Check if lookup_array is provided
        if lookup_array is not None: 
            self.lookup_array = lookup_array
            if verbose:
                print('Using provided lookup_array for accessor.')
            return lookup_array
        
        # Validate dict_of parameter
        if dict_of is None:
            if verbose:
                print('No dict_of provided, defaulting to "dicts" for lookup construction.')
            dict_of = 'dicts'
        
        elif dict_of.lower() not in ('dicts', 'tuples'):
            raise ValueError(f"dict_of must be either 'dicts' or 'tuples', got: {dict_of}")
        
        
        if not hasattr(self, 'lookup_array'):
            
            if verbose: print('Lookup array not found, constructing new lookup array from DataArray coordinates.')

            #==== 1) Global Coordinate Name
            """
            Name of global coordinate dimension in the lookup array. 
            Assumed to be second dimension in the data array, or first dimension not including 'Time' dimension.
            """

            if global_coord_name is None:
                global_coord_name = [dim for dim in self._da.dims if dim.lower() != 'time'][0]
                if verbose:
                    print(f'No global_coord_name provided, inferred as "{global_coord_name}" from DataArray dimensions.')
            
            
            #==== 2) Global Coordinate Values
            """
            Extract list of global coordinates to include in the DataArray.
            """

            global_values= self._da.coords[global_coord_name].values.tolist()
            if verbose:
                print(f'Extracted {len(global_values)} global coordinates from DataArray from dimension "{global_coord_name}" for lookup construction.')
            
            #==== 3) Virtual Coordinate Names
            """
            Names of coordinates that map each global coordinate to its virtual coordinate information.
            Assumed to be all coordinates in the data array except for the global coordinate and 'Time' dimension.
            """
            virtual_coord_names = [ coord for coord in self._da.coords.keys() if coord != global_coord_name and coord.lower() != 'time']
            if verbose:
                print(f'Inferred virtual coordinate names for lookup construction: {virtual_coord_names}')
            
            #==== 4) Build Virtual Map 
            """
            Dictionary mapping global coordinates to their virtual coordinate information. Can be in dict-of-dicts or dict-of-tuples format.
            """

            if dict_of == 'tuples':
                virtual_map = dict(zip(
                    global_values,
                    zip(*(self._da.coords[name].values.tolist() for name in virtual_coord_names))
                ))
            
            elif dict_of == 'dicts':
                virtual_map = {
                    gcoord: {vcoord: self._da.sel({global_coord_name: gcoord})[vcoord].item() for vcoord in virtual_coord_names}
                    for gcoord in global_values
                    }
            if verbose:
                print(f'Constructed virtual_map with {len(virtual_map)} entries in format dict-of-{dict_of} for lookup construction.')
            
            #==== 5) Construct Lookup Array
            self.lookup_array, _ = construct_lookup_array(virtual_map = virtual_map,
                                                          global_coord_name= global_coord_name,
                                                          virtual_coord_names= virtual_coord_names,
                                                          dict_of = dict_of,
                                                          name= f'{self._da.name}_lookup_array' if self._da.name else f'{global_coord_name}_lookup_array',
                                                          verbose= verbose
                                                          )
            if verbose:
                print(f'Constructed lookup_array with dimensions: {self.lookup_array.dims}, coordinates: {self.lookup_array.coords.keys()} and shape: {self.lookup_array.shape}')
        return self.lookup_array
    

    def __call__(self,
                 verbose: bool= False,
                 **selectors
                 ):
        '''
        Select global coordinates based on specified criteria.
        -------------------------------
        Parameters:
            verbose: bool
                If True, function will print additional information during execution. Default is False.
            **selectors:
                Keyword arguments specifying selection criteria for virtual coordinate dimensions. Keys are virtual coordinate names, values can be single values, lists, slices, or None.
        Returns:
            list:
                A list of global coordinates that match the specified selection criteria.
        
        Examples:
            da.ulookup(device='Behavior1')
            da.ulookup(device=slice('Behavior1', 'Behavior3'), localID='DIPort0')
            da.ulookup(verbose=True, device='Behavior1')
        '''
        global_coord_name = [dim for dim in self._da.dims if dim.lower() != 'time'][0]
        return ulookup(lookup_array= self.lookup_array,
                       global_coord_name= global_coord_name,
                       verbose= verbose,
                       **selectors
                       )
        
    def sel(self,
            verbose: bool= False,
            **selectors
            ):
        '''
        Alias for __call__ to allow selection using .sel() syntax.
        -------------------------------        
        
        Parameters:
            verbose: bool
                If True, print detailed information
            **selectors:
                Selection criteria
        
        Returns:
            list: Global coordinates matching the criteria
        
        Examples:
            da.ulookup.sel(device='Behavior1')
            da.ulookup.sel(device='Behavior1', localID='DIPort0')
        '''
        return self.__call__(verbose= verbose, **selectors)

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
    Updates DataArray with values from dfs_dict based on mapping provided in virtual_map. Format of dfs_dict and virtual_map should match, i.e. if dfs_dict is indexable as [global]: [virtual_1][virtual_2].
    --------------------------------
    
    Parameters:
        data_array: xr.DataArray
            The DataArray to be updated. Should have dimensions for time and global coordinates, and virtual coordinate information in its coordinates.
        virtual_map: dict
            A dictionary mapping global coordinates to virtual coordinates. Format should match the structure of dfs_dict for lookups. Can be in dict-of-dicts or dict-of-tuples format.
        dict_of: str
            Specifies the format of virtual_map and dfs_dict. Options are 'dicts' or 'tuples'. Default is 'dicts'.
        dfs_dict: dict
            A nested dictionary containing the source data for updating the DataArray. Should be structured to allow lookups based on the virtual coordinates specified in virtual_map. For example, if virtual_map is dict-of-dicts with virtual coordinates 'device' and 'localID', dfs_dict should be structured as dfs_dict[device][localID][global_coord] to retrieve the value for a given global coordinate.
        fill_value: any
            A value to use for filling entries in the DataArray when lookups fail or when virtual coordinate information is missing. Default is np.nan.
        verbose: bool
            If True, function will print additional information during execution. Default is False.
    
    Returns:
        xr.DataArray
            The updated DataArray with values filled in from dfs_dict based on the mapping in virtual_map
    '''

    #===== Validate Inputs
    if data_array is None:                          raise ValueError("data_array parameter must be provided and cannot be None.")
    if virtual_map is None:                         raise ValueError("virtual_map parameter must be provided and cannot be None.")
    if dict_of.lower() not in ('dicts', 'tuples'):  raise ValueError(f"dict_of must be either 'dicts' or 'tuples', got: {dict_of}")
    if dfs_dict is None:                            raise ValueError("dfs_dict parameter must be provided and cannot be None.")

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
                              dict_of: str= 'tuples',
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
    if verbose: 
        print(f'Warning: construct_channel_type_da is deprecated. Use construct_data_array instead with appropriate parameters.')
    
    return construct_data_array(virtual_map = channels_ref_dict,
                                global_coord_name= 'channel',
                                virtual_coord_names= ['device', 'register', 'localID'],
                                times= times,
                                dict_of= dict_of,
                                name= name,
                                test_values= test_help_values,
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
#
################################################################################

                                



################################################################################
# Construct Channel Lookup DataArray
################################################################################

def construct_channel_lookup_da(channels: list= None,
                                channels_ref_dict: dict= None,
                                dict_of: str= 'tuples',
                                verbose: bool= False
                                ) -> xr.DataArray:
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
    if verbose: 
        print(f'Warning: construct_channel_lookup_da is deprecated. Use construct_lookup_array instead with appropriate parameters.')
    
    lookup_da, lookup_virtual_coords = construct_lookup_array(virtual_map= channels_ref_dict,
                                                              global_coord_name= 'channel',
                                                              virtual_coord_names= ['device', 'register', 'localID'],
                                                              dict_of= dict_of,
                                                              name= 'channel_lookup_da',
                                                              verbose= verbose
                                                                )
    
    return lookup_da, lookup_virtual_coords['device'], lookup_virtual_coords['register'], lookup_virtual_coords['localID']

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

def select_channels_from_C_D_L(lookup_da: xr.DataArray = None,
                               unified_coord_name: str = 'channel',
                               verbose: bool = False,
                               channel: str= None,
                               device: str= None,
                               localID: str= None
                               ) -> list:
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
    if verbose: 
        print(f'Warning: select_channels_from_C_D_L is deprecated. Use ulookup instead with appropriate parameters.')
    
    return ulookup(lookup_array = lookup_da,
                     global_coord_name = unified_coord_name,
                     verbose= verbose,
                     channel = channel,
                     device = device,
                     localID = localID
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
###############################################################################
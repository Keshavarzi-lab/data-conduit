'''
Uitility functions for querying virtual arrays using lookup arrays.
Includes:
- ulookup: Select global coordinates from a lookup array based on specified criteria across virtual coordinate dimensions

''' #noqa            


################################################################################
# Imports
################################################################################
import xarray as xr

################################################################################




################################################################################
# Private Helper Functions
################################################################################
#===============================================================================
# 1| Apply Global Coordinate Selector 
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
          if verbose: 
            print(f'Applied slice selector on {dim_name}: {selector}, resulting shape: {results.shape}')

    #== List-Based Selection
    elif isinstance(selector, (list, tuple)):       #                                                   # noqa
        results= results.sel({dim_name: selector})
        if verbose: 
            print(f'Applied list selector on {dim_name}: {selector}, resulting shape: {results.shape}')
    
    #== Single Value Selection
    else:
        results= results.sel({dim_name: [selector]})
        if verbose: 
            print(f'Applied single value selector on {dim_name}: {selector}, resulting shape: {results.shape}')
    return results
    

#===============================================================================




#===============================================================================
# 2| Apply Virtual Coordinate Selector
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
    ''' # noqa

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
    elif isinstance(selector, (list, tuple)):                                                           # noqa
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
################################################################################





################################################################################
# Select Global Coordinates from Lookup Array
################################################################################

def ulookup(lookup_array: xr.DataArray= None,                       # noqa
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
    '''     # noqa

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
        if verbose: 
            print('No selectors provided (all None), returning all global coordinates.')
        return lookup_array[global_coord_name].values.tolist()
    
    if verbose: 
        print(f'Applying following selectors: {sel}')
    
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
            available= [global_coord_name] + [coord for coord in lookup_array.coords if coord != global_coord_name]
            raise ValueError(f'Coordinate name "{coord_name}" not found in lookup_array coordinates: {available}')
    
    #==== Extract Selected Global Coordinates
    selected = results.coords[global_coord_name].values.tolist()
    if verbose:
        print(f'Selected {len(selected)} {global_coord_name}(s)')

        if len(selected) <= 10:     
            print(f'  Result: {selected}')
        else:                       
            print(f'  Result (first 10): {selected[:10]}...')
    
    return selected

################################################################################

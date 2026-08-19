'''Custom xarray DataArray accessor for lookup operations with auto-construction of lookup arrays.'''
################################################################################
# Imports
################################################################################
import xarray as xr

from data_conduit.actual.core.virtualarrays.base_queries import ulookup
from data_conduit.actual.core.virtualarrays.virtual_arrays_core import construct_lookup_array

################################################################################






################################################################################
# Lookup Accessor with Auto-Construction of Lookup Array
################################################################################

@xr.register_dataarray_accessor('ulookup')
class LookupAccessorConstructor:
    '''Base class for constructing custom xarray DataArray accessors for lookup operations with auto-construction of lookup arrays.'''

    def __init__(self,
                 data_array: xr.DataArray
                 ):
        '''Initialise the accessor with the given DataArray and construct the lookup array if not already present.'''
          
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
        '''Construct or validate the lookup DataArray for the accessor.'''

        # Check if lookup_array is provided
        if lookup_array is not None: 
            self.lookup_array = lookup_array
            if verbose:
                print('Using provided lookup_array for accessor.')
            return lookup_array
        
        # Check for pre-attached lookup array (set by MultiSource), which allows sharing lookup arrays across multiple DataArrays without redundant construction.
        # Does not require lookup_array to be passed in constructor, allowing simple accessor usage like da.ulookup.sel(...) while still benefiting from lookup array construction 
        # and reuse when used within a MultiSource context.

        if '_lookup_array_ref' in self._da.attrs:
            self.lookup_array = self._da.attrs['_lookup_array_ref']
            if verbose:
                print('Using pre-attached lookup array from MultiSource.')
            return self.lookup_array
        # Validate dict_of parameter
        if dict_of is None:
            if verbose:
                print('No dict_of provided, defaulting to "dicts" for lookup construction.')
            dict_of = 'dicts'
        
        elif dict_of.lower() not in ('dicts', 'tuples'):
            raise ValueError(f"dict_of must be either 'dicts' or 'tuples', got: {dict_of}")
        
        
        if not hasattr(self, 'lookup_array'):
            
            if verbose: 
                print('Lookup array not found, constructing new lookup array from DataArray coordinates.')

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
            virtual_coord_names = [ coord for coord in self._da.coords if coord != global_coord_name and coord.lower() != 'time']
            if verbose:
                print(f'Inferred virtual coordinate names for lookup construction: {virtual_coord_names}')
            
            #==== 4) Build Virtual Map 
            """
            Dictionary mapping global coordinates to their virtual coordinate information. Can be in dict-of-dicts or dict-of-tuples format.
            """

            if dict_of == 'tuples':
                virtual_map = dict(zip(
                    global_values,
                    zip(*(self._da.coords[name].values.tolist() for name in virtual_coord_names), strict= True),
                    strict= True
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
                Keyword arguments specifying selection criteria for virtual coordinate dimensions. 
                Keys are virtual coordinate names, values can be single values, lists, slices, or None.
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
        '''Alias for select().'''
        return self.select(verbose= verbose, **selectors)

    def select(self,
               verbose: bool= False,
               **selectors
               ):
        '''
        Return the DataArray sliced to matching global coordinates.
        -------------------------------        
        
        Parameters:
            verbose: bool
                If True, print detailed information
            **selectors:
                Selection criteria
        
        Returns:
            xr.DataArray: DataArray sliced to matching coordinates
        
        Examples:
            da.ulookup.select(device='Behavior1')
            da.ulookup.select(device='Behavior1', localID='DIPort0')
        '''
        coords = self.__call__(verbose= verbose, **selectors)
        global_coord_name = [dim for dim in self._da.dims if dim.lower() != 'time'][0]
        return self._da.sel({global_coord_name: coords})

################################################################################
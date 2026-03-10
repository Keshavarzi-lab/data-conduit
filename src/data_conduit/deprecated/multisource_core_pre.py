'''
MultiSource Base Class.
--------------------------------

Description:
    Source-agnostic orchestrator for constructing virtual DataArrays
    from any DataSource's dfs_dict. Takes a nested dict of DataFrames 
    plus virtual_maps, and runs the full pipeline: timestamp collection,
    base DataArray construction, lookup array construction, and value 
    population.

    MultiSource is to virtual arrays what DataSource is to raw data:
    a base class that subclasses configure for specific data sources
    (e.g. MultiDevice for HARP binary data).

Contents:
--------------------------------
- MultiSource
    Base class for multi-source virtual array construction.
'''


################################################################################
# Imports
################################################################################

from pathlib import Path

import numpy as np
import xarray as xr

from data_conduit.timestamps import collect_timestamps_nested
from data_conduit.virtualarrays import (
    construct_data_array,
    construct_lookup_array,
    update_data_array,
)

################################################################################




################################################################################
# MultiSource Base Class
################################################################################

class MultiSource:
    '''
    Source-agnostic orchestrator for constructing virtual DataArrays
    from a nested dict of DataFrames and virtual coordinate maps.

    Given a dfs_dict (from any DataSource) and virtual_maps describing
    how global coordinates map to positions within that dict, MultiSource
    runs the full pipeline:
        1. Collect timestamps from dfs_dict
        2. Construct base DataArrays (Time × global_coords)
        3. Construct lookup arrays (global_coord → virtual coords)
        4. Populate DataArrays with values from dfs_dict

    Subclasses (e.g. MultiDevice) provide the dfs_dict by creating
    a DataSource internally.

    Parameters
    ----------
    dfs_dict : dict
        Nested dictionary of DataFrames, e.g. from DataSource.dfs_dict.
        Structure must match virtual_maps navigation paths.
        e.g. {device: {register: DataFrame(Time index, localID columns)}}
    virtual_maps : dict[str, dict]
        Mapping of data_keys to virtual maps.
        Each virtual map is {global_coord: {vcoord_name: vcoord_value, ...}}.
        e.g. {'Activations': {'NP0': {'device': 'Behavior0', 'register': '32', 'localID': 'DIPort0'}, ...}}
    data_keys : list[str] or None
        List of data keys to process. If None, uses virtual_maps.keys().
    global_coord_name : str
        Name of the global coordinate dimension. Default 'global_coord'.
    virtual_coord_names : list[str] or None
        Names of virtual coordinate dimensions. If None, inferred from
        the first entry of the first virtual map.
    dict_of : str
        Format of virtual_map entries: 'dicts' or 'tuples'.
    data_array_names : dict[str, str] or None
        Custom names for DataArrays, keyed by data_key.
        Defaults to '{data_key}_data'.
    data_array_attrs : dict or None
        Attributes to attach to DataArrays.
    lookup_array_names : dict[str, str] or None
        Custom names for lookup arrays, keyed by data_key.
        Defaults to '{data_key}_lookup'.
    lookup_array_attrs : dict or None
        Attributes to attach to lookup arrays.
    test_values : bool
        If True, populate DataArrays with human-readable test strings.
    fill_value : any
        Value for missing entries when populating DataArrays.
    verbose : bool
        If True, print progress during construction.

    Attributes
    ----------
    dfs_dict : dict
        The nested dictionary of DataFrames.
    virtual_maps : dict
        The virtual coordinate maps.
    data_arrays : dict[str, xr.DataArray]
        Constructed DataArrays, keyed by data_key.
    lookup_arrays : dict[str, xr.DataArray]
        Constructed lookup arrays, keyed by data_key.
    lookup_virtual_coords : dict[str, dict]
        Unique virtual coordinate values per data_key.
    '''

    def __init__(self,
                 dfs_dict: dict,
                 virtual_maps: dict,
                 data_keys: list | None = None,
                 global_coord_name: str = 'global_coord',
                 virtual_coord_names: list | None = None,
                 dict_of: str = 'dicts',
                 data_array_names: dict | None = None,
                 data_array_attrs: dict | None = None,
                 lookup_array_names: dict | None = None,
                 lookup_array_attrs: dict | None = None,
                 test_values: bool = False,
                 fill_value=None,
                 verbose: bool = False,
                 ):
        '''
        Initialise MultiSource.

        Stores configuration, then runs construct_all to build
        data_arrays and lookup_arrays for all data_keys.
        '''
        self.dfs_dict = dfs_dict
        self.virtual_maps = virtual_maps
        self.global_coord_name = global_coord_name
        self.virtual_coord_names = virtual_coord_names
        self.dict_of = dict_of
        self.data_array_names = data_array_names or {}
        self.data_array_attrs = data_array_attrs
        self.lookup_array_names = lookup_array_names or {}
        self.lookup_array_attrs = lookup_array_attrs
        self.test_values = test_values
        self.fill_value = fill_value
        self.verbose = verbose

        #=== Resolve data_keys
        if data_keys is not None:
            self.data_keys = data_keys
        else:
            self.data_keys = list(virtual_maps.keys())

        #=== Infer virtual_coord_names if not provided
        if self.virtual_coord_names is None:
            self.virtual_coord_names = self._infer_virtual_coord_names()

        #=== Construct all outputs
        if self.dfs_dict is not None and self.virtual_maps is not None:
            self._construct_all()
        else:
            self.data_arrays = {}
            self.lookup_arrays = {}
            self.lookup_virtual_coords = {}
            if verbose:
                print("Skipping auto-construction: dfs_dict or "
                      "virtual_maps not provided.")


    #===========================================================================
    # Infer Virtual Coordinate Names
    #===========================================================================

    def _infer_virtual_coord_names(self) -> list[str]:
        '''
        Infer virtual coordinate names from the first entry of the 
        first virtual map.

        Returns
        -------
        list[str]
            Virtual coordinate names.
        '''
        first_key = self.data_keys[0]
        first_vmap = self.virtual_maps[first_key]
        first_entry = next(iter(first_vmap.values()))

        if self.dict_of == 'dicts':
            return list(first_entry.keys())
        else:
            raise ValueError(
                "Cannot infer virtual_coord_names from tuples format. "
                "Please provide virtual_coord_names explicitly."
            )


    #===========================================================================
    # Construct Outputs for a Single Data Key
    #===========================================================================

    def construct_data_outputs(self,
                               data_key: str,
                               dfs_dict: dict | None = None,
                               virtual_map: dict | None = None,
                               data_array_name: str | None = None,
                               da_attrs: dict | None = None,
                               lookup_array_name: str | None = None,
                               lookup_attrs: dict | None = None,
                               fill_value=None,
                               verbose: bool | None = None,
                               ) -> tuple[xr.DataArray, xr.DataArray, dict]:
        '''
        Construct data_array, lookup_array for a single data_key.

        Steps:
            1. Resolve virtual_map for this data_key.
            2. Collect timestamps from dfs_dict.
            3. Construct base DataArray.
            4. Construct lookup array.
            5. Populate DataArray with values from dfs_dict.

        Parameters
        ----------
        data_key : str
            The data key to process (e.g. 'Activations').
        dfs_dict : dict or None
            Override dfs_dict. Falls back to self.dfs_dict.
        virtual_map : dict or None
            Override virtual_map. Falls back to self.virtual_maps[data_key].
        data_array_name : str or None
            Name for the DataArray. Defaults to '{data_key}_data'.
        da_attrs : dict or None
            Attributes for the DataArray.
        lookup_array_name : str or None
            Name for the lookup array. Defaults to '{data_key}_lookup'.
        lookup_attrs : dict or None
            Attributes for the lookup array.
        fill_value : any or None
            Fill value. Falls back to self.fill_value.
        verbose : bool or None
            Falls back to self.verbose.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray, dict]
            (data_array, lookup_array, lookup_virtual_coords)
        '''
        if verbose is None:
            verbose = self.verbose
        if dfs_dict is None:
            dfs_dict = self.dfs_dict
        if fill_value is None:
            fill_value = self.fill_value

        #=== 1. Resolve virtual_map
        if virtual_map is None:
            if data_key not in self.virtual_maps:
                raise KeyError(
                    f"data_key '{data_key}' not in virtual_maps. "
                    f"Available: {list(self.virtual_maps.keys())}"
                )
            virtual_map = self.virtual_maps[data_key]

        if verbose:
            print(f"--- construct_data_outputs: data_key='{data_key}' ---")

        #=== 2. Collect timestamps
        if verbose:
            print("Step 1: Collecting timestamps...")
        timestamps = collect_timestamps_nested(
            dfs_dict=dfs_dict,
            return_type='list',
            verbose=verbose,
        )

        #=== 3. Construct base DataArray
        if data_array_name is None:
            data_array_name = self.data_array_names.get(
                data_key, f'{data_key}_data'
            )
        if da_attrs is None:
            da_attrs = self.data_array_attrs

        if verbose:
            print(f"Step 2: Constructing base DataArray '{data_array_name}'...")
        data_array = construct_data_array(
            virtual_map=virtual_map,
            global_coord_name=self.global_coord_name,
            virtual_coord_names=self.virtual_coord_names,
            dict_of=self.dict_of,
            times=timestamps,
            name=data_array_name,
            test_values=self.test_values,
            verbose=verbose,
            da_attrs=da_attrs,
        )

        #=== 4. Construct lookup array
        if lookup_array_name is None:
            lookup_array_name = self.lookup_array_names.get(
                data_key, f'{data_key}_lookup'
            )
        if lookup_attrs is None:
            lookup_attrs = self.lookup_array_attrs

        if verbose:
            print(f"Step 3: Constructing lookup array '{lookup_array_name}'...")
        lookup_da, lookup_virtual_coords = construct_lookup_array(
            virtual_map=virtual_map,
            global_coord_name=self.global_coord_name,
            virtual_coord_names=self.virtual_coord_names,
            dict_of=self.dict_of,
            name=lookup_array_name,
            verbose=verbose,
            lookup_attrs=lookup_attrs,
        )

        #=== 5. Update DataArray with values
        if verbose:
            print("Step 4: Updating DataArray with values from dfs_dict...")
        data_array = update_data_array(
            data_array=data_array,
            virtual_map=virtual_map,
            dict_of=self.dict_of,
            dfs_dict=dfs_dict,
            fill_value=fill_value if fill_value is not None else np.nan,
            verbose=verbose,
        )

        if verbose:
            print(f"--- construct_data_outputs complete for "
                  f"data_key='{data_key}' ---")

        return data_array, lookup_da, lookup_virtual_coords


    #===========================================================================
    # Construct Outputs for All Data Keys
    #===========================================================================

    def _construct_all(self):
        '''
        Construct data_arrays, lookup_arrays, and lookup_virtual_coords
        for all data_keys.

        Stores results in self.data_arrays, self.lookup_arrays,
        self.lookup_virtual_coords.
        '''
        if self.verbose:
            print(f"Constructing outputs for data_keys: {self.data_keys}")

        self.data_arrays = {}
        self.lookup_arrays = {}
        self.lookup_virtual_coords = {}

        for dk in self.data_keys:
            if self.verbose:
                print(f"\n{'=' * 60}")

            da, lookup_da, lookup_vc = self.construct_data_outputs(
                data_key=dk,
                verbose=self.verbose,
            )
            self.data_arrays[dk] = da
            self.lookup_arrays[dk] = lookup_da
            self.lookup_virtual_coords[dk] = lookup_vc

        if self.verbose:
            print(f"\nAll outputs constructed for {len(self.data_keys)} "
                  f"data_keys.")


################################################################################
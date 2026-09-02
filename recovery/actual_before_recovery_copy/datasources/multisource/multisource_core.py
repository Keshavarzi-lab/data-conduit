"""
MultiSource Base Class.
-----------------------

Description:
    Base class for combined / virtual-array data loading in data-conduit.
    Mirrors MonoSource in the first stage: obtain / build a dfs_dict, or
    accept one directly. In the second stage, builds combined outputs from
    that dfs_dict using virtual maps rather than direct path lookups.

Contents:
-----------------------
- MultiSource
    Base class. Resolves dfs_dict, stores virtual map configuration, and
    builds data_arrays, lookup_arrays, and lookup_virtual_coords.
"""

################################################################################
# Imports
################################################################################

from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr

from ...core.timestamps import collect_timestamps_nested
from ...core.virtualarrays import (
    construct_data_array,
    construct_lookup_array,
    update_data_array,
)
from ..monosource.monosource_core import _obtain_dfs_dict

################################################################################


# ===============================================================================
# 1| Infer virtual coordinate names
# ===============================================================================


def _infer_virtual_coord_names(
    data_keys: list[str],
    virtual_maps: dict,
    dict_of: str,
) -> list[str]:
    """
    Infer virtual coordinate names from the first entry of the first virtual map.

    Parameters
    ----------
    data_keys : list[str]
        Ordered data keys to inspect.
    virtual_maps : dict
        Mapping of data keys to virtual maps.
    dict_of : str
        Format of virtual map entries.

    Returns
    -------
    list[str]
        Inferred virtual coordinate names.
    """
    if not data_keys:
        return []

    first_key = data_keys[0]
    first_vmap = virtual_maps[first_key]
    first_entry = next(iter(first_vmap.values()))

    if dict_of == "dicts":
        return list(first_entry.keys())

    raise ValueError("Cannot infer virtual_coord_names from tuples format. Please provide virtual_coord_names explicitly.")


# ===============================================================================


# ===============================================================================
# 2| Construct outputs for one data_key
# ===============================================================================


def construct_data_outputs(
    data_key: str,
    dfs_dict: dict,
    virtual_maps: dict,
    global_coord_name: str,
    virtual_coord_names: list[str],
    dict_of: str,
    data_array_names: dict,
    data_array_attrs: dict | None,
    lookup_array_names: dict,
    lookup_array_attrs: dict | None,
    test_values: bool,
    fill_value,
    verbose: bool,
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """
    Construct a data array and lookup array for one data key.

    Parameters
    ----------
    data_key : str
        Data key to build outputs for.
    dfs_dict : dict
        Nested dictionary of DataFrames/DataArrays.
    virtual_maps : dict
        Mapping of data keys to virtual maps.
    global_coord_name : str
        Name of the global coordinate dimension.
    virtual_coord_names : list[str]
        Names of the virtual coordinate dimensions.
    dict_of : str
        Format of virtual map entries.
    data_array_names : dict
        Optional custom names for data arrays.
    data_array_attrs : dict or None
        Optional attrs applied to the data array.
    lookup_array_names : dict
        Optional custom names for lookup arrays.
    lookup_array_attrs : dict or None
        Optional attrs applied to the lookup array.
    test_values : bool
        If True, populate arrays with readable test values.
    fill_value : any
        Fill value used for missing entries.
    verbose : bool
        If True, print progress output.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, dict]
        data_array, lookup_array, lookup_virtual_coords.
    """
    if data_key not in virtual_maps:
        raise KeyError(f"data_key '{data_key}' not in virtual_maps. Available: {list(virtual_maps.keys())}")

    virtual_map = virtual_maps[data_key]

    if verbose:
        print(f"--- construct_data_outputs: data_key='{data_key}' ---")
        print("Step 1: Collecting timestamps...")

    timestamps = collect_timestamps_nested(
        dfs_dict=dfs_dict,
        return_type="list",
        verbose=verbose,
    )

    data_array_name = data_array_names.get(data_key, f"{data_key}_data")

    if verbose:
        print(f"Step 2: Constructing base DataArray '{data_array_name}'...")

    data_array = construct_data_array(
        virtual_map=virtual_map,
        global_coord_name=global_coord_name,
        virtual_coord_names=virtual_coord_names,
        dict_of=dict_of,
        times=timestamps,
        name=data_array_name,
        test_values=test_values,
        verbose=verbose,
        da_attrs=data_array_attrs,
    )

    lookup_array_name = lookup_array_names.get(data_key, f"{data_key}_lookup")

    if verbose:
        print(f"Step 3: Constructing lookup array '{lookup_array_name}'...")

    lookup_array, lookup_virtual_coords = construct_lookup_array(
        virtual_map=virtual_map,
        global_coord_name=global_coord_name,
        virtual_coord_names=virtual_coord_names,
        dict_of=dict_of,
        name=lookup_array_name,
        verbose=verbose,
        lookup_attrs=lookup_array_attrs,
    )

    if verbose:
        print("Step 4: Updating DataArray with values from dfs_dict...")

    data_array = update_data_array(
        data_array=data_array,
        virtual_map=virtual_map,
        dict_of=dict_of,
        dfs_dict=dfs_dict,
        fill_value=fill_value if fill_value is not None else np.nan,
        verbose=verbose,
    )

    if verbose:
        print(f"--- construct_data_outputs complete for data_key='{data_key}' ---")

    return data_array, lookup_array, lookup_virtual_coords


# ===============================================================================


# ===============================================================================
# 3| Construct outputs for all data_keys
# ===============================================================================


def construct_all(
    dfs_dict: dict,
    data_keys: list[str],
    virtual_maps: dict,
    global_coord_name: str,
    virtual_coord_names: list[str],
    dict_of: str,
    data_array_names: dict,
    data_array_attrs: dict | None,
    lookup_array_names: dict,
    lookup_array_attrs: dict | None,
    test_values: bool,
    fill_value,
    verbose: bool,
) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray], dict[str, dict]]:
    """
    Construct data arrays, lookup arrays, and lookup virtual coords for all data keys.

    Parameters
    ----------
    dfs_dict : dict
        Nested dictionary of DataFrames/DataArrays.
    data_keys : list[str]
        Data keys to process.
    virtual_maps : dict
        Mapping of data keys to virtual maps.
    global_coord_name : str
        Name of the global coordinate dimension.
    virtual_coord_names : list[str]
        Names of the virtual coordinate dimensions.
    dict_of : str
        Format of virtual map entries.
    data_array_names : dict
        Optional custom names for data arrays.
    data_array_attrs : dict or None
        Optional attrs applied to data arrays.
    lookup_array_names : dict
        Optional custom names for lookup arrays.
    lookup_array_attrs : dict or None
        Optional attrs applied to lookup arrays.
    test_values : bool
        If True, populate arrays with readable test values.
    fill_value : any
        Fill value used for missing entries.
    verbose : bool
        If True, print progress output.

    Returns
    -------
    tuple[dict[str, xr.DataArray], dict[str, xr.DataArray], dict[str, dict]]
        data_arrays, lookup_arrays, lookup_virtual_coords.
    """
    if verbose:
        print(f"Constructing outputs for data_keys: {data_keys}")

    data_arrays: dict[str, xr.DataArray] = {}
    lookup_arrays: dict[str, xr.DataArray] = {}
    lookup_virtual_coords: dict[str, dict] = {}

    for data_key in data_keys:
        if verbose:
            print(f"\n{'=' * 60}")

        data_array, lookup_array, lookup_vc = construct_data_outputs(
            data_key=data_key,
            dfs_dict=dfs_dict,
            virtual_maps=virtual_maps,
            global_coord_name=global_coord_name,
            virtual_coord_names=virtual_coord_names,
            dict_of=dict_of,
            data_array_names=data_array_names,
            data_array_attrs=data_array_attrs,
            lookup_array_names=lookup_array_names,
            lookup_array_attrs=lookup_array_attrs,
            test_values=test_values,
            fill_value=fill_value,
            verbose=verbose,
        )
        data_arrays[data_key] = data_array
        lookup_arrays[data_key] = lookup_array
        lookup_virtual_coords[data_key] = lookup_vc

    if verbose:
        print(f"\nAll outputs constructed for {len(data_keys)} data_keys.")

    return data_arrays, lookup_arrays, lookup_virtual_coords


# ===============================================================================


################################################################################
# MultiSource Base Class
################################################################################


class MultiSource:
    """
    Base class for combined / virtual-array outputs in data-conduit.

    MultiSource mirrors the top-level flow of MonoSource. It either accepts
    a pre-built dfs_dict or builds one via collect_dfs, then constructs
    combined outputs by aligning multiple inputs through virtual maps.

    Parameters
    ----------
    dfs_dict : dict or None
        Optional pre-built nested dictionary of DataFrames/DataArrays. If
        provided, collect_dfs is not called and this dict is used directly.
    experiment_directory_path : str or Path
        Root directory to walk when dfs_dict is not provided.
    readers : dict[str, Callable] or None
        Mapping of file extensions to reader functions.
    reader_kwargs : dict[str, dict] or None
        Mapping of file extensions to kwargs for each reader.
    keep_empty : bool
        If True, preserve empty subdirectories as empty dicts.
    flatten : bool
        If True, flatten the output dict from collect_dfs.
    separator : str
        Separator used when flatten=True.
    virtual_maps : dict or None
        Mapping of data_keys to virtual maps.
    data_keys : list[str] or None
        Data keys to process. If None, uses virtual_maps.keys().
    global_coord_name : str
        Name of the global coordinate dimension.
    virtual_coord_names : list[str] or None
        Names of virtual coordinate dimensions. If None, inferred from the
        first entry of the first virtual map when possible.
    dict_of : str
        Format of virtual_map entries: 'dicts' or 'tuples'.
    data_array_names : dict or None
        Optional custom names for data arrays keyed by data_key.
    data_array_attrs : dict or None
        Optional attrs applied to data arrays.
    lookup_array_names : dict or None
        Optional custom names for lookup arrays keyed by data_key.
    lookup_array_attrs : dict or None
        Optional attrs applied to lookup arrays.
    test_values : bool
        If True, populate data arrays with readable test values.
    fill_value : any
        Fill value used for missing entries in populated arrays.
    verbose : bool
        If True, print progress during loading and construction.
    **kwargs
        Level selectors passed directly to collect_dfs.

    Attributes
    ----------
    dfs_dict : dict
        Nested dictionary of DataFrames/DataArrays.
    virtual_maps : dict
        Virtual map configuration keyed by data_key.
    data_keys : list[str]
        Data keys to construct.
    data_arrays : dict[str, xr.DataArray]
        Constructed virtual data arrays.
    lookup_arrays : dict[str, xr.DataArray]
        Constructed lookup arrays.
    lookup_virtual_coords : dict[str, dict]
        Unique virtual coordinate values for each data key.
    """

    def __init__(
        self,
        dfs_dict: dict | None = None,
        experiment_directory_path: str | Path | None = None,
        readers: dict[str, Callable] | None = None,
        reader_kwargs: dict[str, dict] | None = None,
        keep_empty: bool = False,
        flatten: bool = False,
        separator: str = ":",
        virtual_maps: dict | None = None,
        data_keys: list[str] | None = None,
        global_coord_name: str = "global_coord",
        virtual_coord_names: list[str] | None = None,
        dict_of: str = "dicts",
        data_array_names: dict | None = None,
        data_array_attrs: dict | None = None,
        lookup_array_names: dict | None = None,
        lookup_array_attrs: dict | None = None,
        test_values: bool = False,
        fill_value=None,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialise MultiSource and build outputs when virtual_maps are provided."""
        super(MultiSource, self).__init__()  # noqa

        self.verbose = verbose
        self.virtual_maps = virtual_maps or {}
        self.global_coord_name = global_coord_name
        self.virtual_coord_names = virtual_coord_names
        self.dict_of = dict_of
        self.data_array_names = data_array_names or {}
        self.data_array_attrs = data_array_attrs
        self.lookup_array_names = lookup_array_names or {}
        self.lookup_array_attrs = lookup_array_attrs
        self.test_values = test_values
        self.fill_value = fill_value

        self.dfs_dict = _obtain_dfs_dict(
            dfs_dict=dfs_dict,
            experiment_directory_path=experiment_directory_path,
            readers=readers,
            reader_kwargs=reader_kwargs,
            keep_empty=keep_empty,
            flatten=flatten,
            separator=separator,
            verbose=verbose,
            **kwargs,
        )

        if data_keys is not None:
            self.data_keys = data_keys
        else:
            self.data_keys = list(self.virtual_maps.keys())

        self.data_arrays: dict[str, xr.DataArray] = {}
        self.lookup_arrays: dict[str, xr.DataArray] = {}
        self.lookup_virtual_coords: dict[str, dict] = {}

        if not self.virtual_maps:
            return

        if self.virtual_coord_names is None:
            self.virtual_coord_names = _infer_virtual_coord_names(
                data_keys=self.data_keys,
                virtual_maps=self.virtual_maps,
                dict_of=self.dict_of,
            )

        (
            self.data_arrays,
            self.lookup_arrays,
            self.lookup_virtual_coords,
        ) = construct_all(
            dfs_dict=self.dfs_dict,
            data_keys=self.data_keys,
            virtual_maps=self.virtual_maps,
            global_coord_name=self.global_coord_name,
            virtual_coord_names=self.virtual_coord_names or [],
            dict_of=self.dict_of,
            data_array_names=self.data_array_names,
            data_array_attrs=self.data_array_attrs,
            lookup_array_names=self.lookup_array_names,
            lookup_array_attrs=self.lookup_array_attrs,
            test_values=self.test_values,
            fill_value=self.fill_value,
            verbose=self.verbose,
        )

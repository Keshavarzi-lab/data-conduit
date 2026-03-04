'''
DataSource Base Class.
--------------------------------

Description:
    Base class for all data sources in data-conduit. Wraps collect_dfs 
    to load data from a directory tree into a nested dictionary of 
    DataFrames, with optional named DataArray access via name_map.

Contents:
--------------------------------
- DataSource
    Base class. Calls collect_dfs, stores dfs_dict, builds data_arrays 
    from name_map.
'''


################################################################################
# Imports
################################################################################

from pathlib import Path
from collections.abc import Callable

import pandas as pd
import xarray as xr

from data_conduit.io import collect_dfs

################################################################################




################################################################################
# DataSource Base Class
################################################################################

class DataSource:
    '''
    Base class for data sources in data-conduit.

    Calls collect_dfs to load data, then optionally builds named 
    DataArrays by navigating dfs_dict using name_map.

    Subclasses configure what to load by setting readers, reader_kwargs,
    and level selectors. For example, HarpDevice sets readers={'.bin': 
    read_harp_bin} and FileTypeData sets readers={'.csv': read_csv}.

    Parameters
    ----------
    experiment_directory_path : str or Path
        Root directory to walk.
    readers : dict[str, Callable] or None
        Mapping of file extensions to reader functions.
        e.g. {'.bin': read_harp_bin, '.csv': read_csv}
    reader_kwargs : dict[str, dict] or None
        Mapping of file extensions to kwargs for each reader.
        e.g. {'.bin': {'harp_device_yaml_path': 'device.yml'}}
    name_map : dict or None
        Mapping of friendly names to paths in dfs_dict.
        Each value is a tuple of keys navigating the nested dict.
        The DataFrame found there is converted to xr.DataArray.
        e.g. {'PlaySoundFreq': ('SoundCard', '32')}
    keep_empty : bool
        If True, preserve empty subdirectories as empty dicts.
    flatten : bool
        If True, flatten the output dict.
    separator : str
        Separator for flattened keys.
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Level selectors passed directly to collect_dfs.
        e.g. l0_selector=['Behavior0'], l2_selector=lambda k: ...

    Attributes
    ----------
    dfs_dict : dict
        Nested dictionary of DataFrames from collect_dfs.
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from name_map.
    '''

    def __init__(self,
                 experiment_directory_path: str | Path,
                 readers: dict[str, Callable] | None = None,
                 reader_kwargs: dict[str, dict] | None = None,
                 name_map: dict | None = None,
                 keep_empty: bool = False,
                 flatten: bool = False,
                 separator: str = ':',
                 verbose: bool = False,
                 **kwargs,
                 ):
        '''
        Initialise DataSource.
        Calls collect_dfs to load data, then optionally builds named 
        DataArrays by navigating dfs_dict using name_map. Subclasses configure what to load by setting readers, reader_kwargs, and level selectors.
        '''
        self.experiment_directory_path = Path(experiment_directory_path)
        self.verbose = verbose

        #=== i| Load Data
        self.dfs_dict = collect_dfs(
            base_path=experiment_directory_path,
            readers=readers,
            reader_kwargs=reader_kwargs,
            keep_empty=keep_empty,
            flatten=flatten,
            separator=separator,
            verbose=verbose,
            **kwargs,
        )

        #=== ii| Build Named DataArrays
        self.data_arrays = {}
        if name_map is not None:
            self._build_data_arrays(name_map)


    def _build_data_arrays(self, name_map: dict):
        '''
        Navigate dfs_dict using name_map and convert to DataArrays.

        Parameters
        ----------
        name_map : dict
            Mapping of {friendly_name: (key1, key2, ...)} where the 
            keys navigate the nested dfs_dict to a DataFrame.
            e.g. {'PlaySoundFreq': ('SoundCard', '32')}
        '''
        for name, path in name_map.items():
            if isinstance(path, str):
                path = (path,)
            try:
                current = self.dfs_dict
                for key in path:
                    current = current[key]
                if isinstance(current, pd.DataFrame):
                    self.data_arrays[name] = current.to_xarray()
                elif isinstance(current, xr.DataArray):
                    self.data_arrays[name] = current
                else:
                    if self.verbose:
                        print(f"Warning: '{name}' at {path} is "
                              f"{type(current).__name__}, not DataFrame.")
            except KeyError as e:
                if self.verbose:
                    print(f"Warning: '{name}' not found at {path}: {e}")


################################################################################
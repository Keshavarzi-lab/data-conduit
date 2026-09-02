'''
MonoSource Base Class.
-----------------------

Description:
    Base class for all data sources in data-conduit. Wraps collect_dfs 
    to load data from a directory tree into a nested dictionary of 
    DataFrames, with optional named DataArray access via monosource_data_arrays.

Contents:
-----------------------
- MonoSource
    Base class. Calls collect_dfs, stores dfs_dict, builds data_arrays 
    from monosource_data_arrays.
'''

################################################################################
# Imports
################################################################################

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import xarray as xr

from data_conduit.core.io import collect_dfs
from data_conduit.core.utils import _apply_level_selectors, _flatten_nested_dict, _parse_selectors

################################################################################






#===============================================================================
# 1| Obtain dfs_dict 
#===============================================================================

def _obtain_dfs_dict(
    dfs_dict: dict | None,
    experiment_directory_path: str | Path | None,
    readers: dict[str, Callable] | None,
    reader_kwargs: dict[str, dict] | None,
    keep_empty: bool,
    flatten: bool,
    separator: str,
    verbose: bool,
    **kwargs,
) -> dict:
    '''
    Obtain dfs_dict for MonoSource/MultiSource.

    If dfs_dict is provided, validate it and return it directly.
    If dfs_dict is not provided, check that experiment_directory_path and other
    required arguments are provided, then call collect_dfs with the provided
    args to build dfs_dict.

    Parameters
    ----------
    dfs_dict : dict or None
        Optional pre-built nested dictionary of DataFrames. If provided, 
        collect_dfs is not called and this dict is used directly.
    experiment_directory_path : str or Path
        Root directory to walk.
    readers : dict[str, Callable] or None
        Mapping of file extensions to reader functions.
        e.g. {'.bin': read_harp_bin, '.csv': read_csv}
    reader_kwargs : dict[str, dict] or None
        Mapping of file extensions to kwargs for each reader.
        e.g. {'.bin': {'harp_device_yaml_path': 'device.yml'}}
    keep_empty: bool
        If True, preserve empty subdirectories in dfs_dict as empty dicts. 
        If False, skip them.
    flatten: bool
        If True, flatten the output dict from collect_dfs. If False, keep the
        nested structure.
        e.g. flatten=True turns {'a': {'b': df}} into {'a/b': df}.
    separator: str
        If flatten=True, the separator to use when joining nested keys.
        e.g. separator='/' turns {'a': {'b': df}} into {'a/b': df}.
    verbose : bool
        If True, print verbose output during collect_dfs.
    **kwargs
        Level selectors passed directly to collect_dfs. 
        e.g. l0_selector=['Behavior0'], l2_selector=lambda k: ...   

    Returns
    -------
    dict
        Nested dictionary of DataFrames from collect_dfs or provided directly.
    '''

    if dfs_dict is not None:
        if not isinstance(dfs_dict, dict):
            raise TypeError(
                f"dfs_dict must be a dict if provided, got {type(dfs_dict).__name__}."
            )
        return dfs_dict

    if experiment_directory_path is None:
        raise ValueError(
            "Provide either dfs_dict or experiment_directory_path so collect_dfs can build one."
        )

    return collect_dfs(
        base_path =experiment_directory_path,
        readers=readers,
        reader_kwargs=reader_kwargs,
        keep_empty=keep_empty,
        flatten=flatten,
        separator=separator,
        verbose=verbose,
        **kwargs,
    )


#===============================================================================







#===============================================================================
# 2| Build data_arrays from monosource_data_arrays 
#===============================================================================

def _build_data_arrays(
    dfs_dict: dict,
    monosource_data_arrays: dict,
    verbose: bool = False,
) -> dict[str, xr.DataArray]:
    """
    Build named xarray objects from a nested dfs_dict using level selectors.

    Each entry in monosource_data_arrays maps a friendly name to a dict of
    level selectors, using the same ``l{n}_selector`` format as collect_dfs.

    Parameters
    ----------
    dfs_dict : dict
        Nested dictionary of DataFrames/DataArrays.
    monosource_data_arrays : dict
        Mapping of {friendly_name: selector_dict}.
        
    Examples:
            {'PlaySoundFreq': {'l0_selector': 'SoundCard', 'l1_selector': '32'}}
            {'Everything': {}}
    verbose : bool
        If True, print warnings for missing or invalid paths.

    Returns
    -------
    dict[str, xr.DataArray]
        Named DataArrays. Single matches are stored under the friendly name.
        Multiple matches are stored as '{name}:{flattened_key}'.
    """
    data_arrays: dict[str, xr.DataArray] = {}

    for name, selectors in monosource_data_arrays.items():
        level_selectors = _parse_selectors(selectors)
        filtered = _apply_level_selectors(dfs_dict, level_selectors)
        flat = _flatten_nested_dict(filtered)

        if len(flat) == 0 and verbose:
            print(f"Warning: '{name}' matched nothing in dfs_dict.")
            continue

        for flat_key, leaf in flat.items():
            array_name = name if len(flat) == 1 else f"{name}:{flat_key}"
            if isinstance(leaf, pd.DataFrame):
                data_arrays[array_name] = leaf.to_xarray()
            elif isinstance(leaf, xr.DataArray):
                data_arrays[array_name] = leaf
            elif verbose:
                print(f"Warning: '{array_name}' is {type(leaf).__name__}, not DataFrame/DataArray.")

    return data_arrays
#===============================================================================






################################################################################
# MonoSource Base Class
################################################################################

class MonoSource:
    '''
    Base class for data sources in data-conduit.

    Calls collect_dfs to load data, then optionally builds named 
    DataArrays by navigating dfs_dict using monosource_data_arrays. If
    dfs_dict is provided, it is used directly and collect_dfs is not called.

    Subclasses configure what to load by setting readers, reader_kwargs,
    and level selectors. For example, HarpDevice sets readers={'.bin':
    read_harp_bin} and FileTypeData sets readers={'.csv': read_csv}.

    Parameters
    ----------
    dfs_dict : dict or None
        Optional pre-built nested dictionary of DataFrames. If provided, 
        collect_dfs is not called and this dict is used directly.
    experiment_directory_path : str or Path
        Root directory to walk.
    readers : dict[str, Callable] or None
        Mapping of file extensions to reader functions.
        e.g. {'.bin': read_harp_bin, '.csv': read_csv}
    reader_kwargs : dict[str, dict] or None
        Mapping of file extensions to kwargs for each reader.
        e.g. {'.bin': {'harp_device_yaml_path': 'device.yml'}}
    keep_empty: bool
        If True, preserve empty subdirectories in dfs_dict as empty dicts. 
        If False, skip them.
    flatten: bool
        If True, flatten the output dict from collect_dfs. If False, keep the
        nested structure.
        e.g. flatten=True turns {'a': {'b': df}} into {'a/b': df}.
    separator: str
        If flatten=True, the separator to use when joining nested keys.
        e.g. separator='/' turns {'a': {'b': df}} into {'a/b': df}.
    monosource_data_arrays : dict or None
        Optional mapping of DataArray names to paths in dfs_dict. If provided,
        these DataArrays are built from dfs_dict and stored as attributes for
        easy access. 
        Each value is a tuple of keys navigating the nested dict to find the
        DataFrame to convert to a DataArray.
        e.g. {'PlaySoundFreq': ('SoundCard', '32')} for 
        dfs_dict['SoundCard']['32'] -> DataFrame -> DataArray named 'PlaySoundFreq'
    verbose : bool
        If True, print verbose output during collect_dfs.
    **kwargs
        Level selectors passed directly to collect_dfs. 
        e.g. l0_selector=['Behavior0'], l2_selector=lambda k: ...
    
    Attributes
    ----------
    dfs_dict : dict
        Nested dictionary of DataFrames from collect_dfs or provided directly.
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from monosource_data_arrays.
    '''

    _verbose_disabled_warning_shown_for: set[type] = set()                          # 

    def __init__(
            self,
            dfs_dict: dict | None = None,
            #=== collect_dfs args ===#
            experiment_directory_path: str | Path | None = None,
            readers: dict[str, Callable] | None = None,
            reader_kwargs: dict[str, dict] | None = None,
            keep_empty: bool = False,
            flatten: bool = False,
            separator: str = ':',
            verbose: bool = False,
            #=== monosource_data_arrays args ===#
            monosource_data_arrays: dict | None = None,
            #=== collect_dfs level selectors passed as kwargs ===#
            **kwargs,
    ):
        '''
        Initialise MonoSource.

        Calls collect_dfs to load data, then optionally builds named 
        DataArrays by navigating dfs_dict using monosource_data_arrays. If
        dfs_dict is provided, it is used directly and collect_dfs is not called.

        Subclasses configure what to load by setting readers, reader_kwargs,
        and level selectors. For example, HarpDevice sets readers={'.bin':
        read_harp_bin} and FileTypeData sets readers={'.csv': read_csv}.

        Parameters
        ----------
        dfs_dict : dict or None
            Optional pre-built nested dictionary of DataFrames. If provided, 
            collect_dfs is not called and this dict is used directly.
        experiment_directory_path : str or Path
            Root directory to walk.
        readers : dict[str, Callable] or None
            Mapping of file extensions to reader functions.
            e.g. {'.bin': read_harp_bin, '.csv': read_csv}
        reader_kwargs : dict[str, dict] or None
            Mapping of file extensions to kwargs for each reader.
            e.g. {'.bin': {'harp_device_yaml_path': 'device.yml'}}
        keep_empty: bool
            If True, preserve empty subdirectories in dfs_dict as empty dicts. 
            If False, skip them.
        flatten: bool
            If True, flatten the output dict from collect_dfs. If False, keep the
            nested structure.
            e.g. flatten=True turns {'a': {'b': df}} into {'a/b': df}.
        separator: str
            If flatten=True, the separator to use when joining nested keys.
            e.g. separator='/' turns {'a': {'b': df}} into {'a/b': df}.
        monosource_data_arrays : dict or None
            Optional mapping of DataArray names to paths in dfs_dict. If provided,
            these DataArrays are built from dfs_dict and stored as attributes for
            easy access. 
            Each value is a tuple of keys navigating the nested dict to find the
            DataFrame to convert to a DataArray.
            e.g. {'PlaySoundFreq': ('SoundCard', '32')} for 
            dfs_dict['SoundCard']['32'] -> DataFrame -> DataArray named 'PlaySoundFreq'
        verbose : bool
            If True, print verbose output during collect_dfs.
        **kwargs
            Level selectors passed directly to collect_dfs. 
            e.g. l0_selector=['Behavior0'], l2_selector=lambda k: ...   
        '''

        ''' 
        Components:

        1. Check if dfs_dict is provided.
            - If yes, validate it and use it directly.
            - If no, check that experiment_directory_path and other required arguments are provided, then call
              collect_dfs with the provided args to build dfs_dict.
        2. If monosource_data_arrays is provided, build each DataArray by navigating dfs_dict using the provided paths.
            - If any path is invalid, raise warning and skip that DataArray.
        3. Store dfs_dict and data_arrays as attributes.
        '''

        super(MonoSource, self).__init__()  # noqa

        self.verbose = verbose
        ms_class = self.__class__       # Store the class of the current instance for verbose warning checks
        
        # Show warning if verbose is disabled for this class without repeating it for subclasses. 
        # This avoids spamming the user with warnings if they have multiple monosource classes.
        if not verbose and ms_class not in self._verbose_disabled_warning_shown_for:                
            print(f'''
                  Warning: verbose output disabled for {ms_class.__name__}. 
                  If any files/paths are missing or invalid for the monosource_data_arrays you specified, 
                  you may not see warnings about them. Set verbose=True to enable warnings.
                  ''')
            self._verbose_disabled_warning_shown_for.add(ms_class)
        
     
        #=== 1. Obtain dfs_dict 

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

        #== 2. Build data_arrays from monosource_data_arrays
        self.data_arrays = {}
        if monosource_data_arrays is not None:
            self.data_arrays = _build_data_arrays(
                dfs_dict=self.dfs_dict,
                monosource_data_arrays=monosource_data_arrays,
                verbose= self.verbose,
            )



'''
FileTypeData and preset subclasses.
--------------------------------

Description:
    FileTypeData extends MonoSource for common file-type data (CSV, JSON,
    JSONL, YAML). Analogous to Device but for flat-file formats.
    Handles device_type folder filtering, reader resolution from file_type
    strings, and post-load column/index renaming.

    Subclasses provide preset configurations for specific data sources
    (ExperimentEvents, RotationData, VideoData, etc.), each setting
    defaults for device_type, file_type, renaming, and validation.

Contents:
--------------------------------
- FileTypeData(MonoSource)
    Intermediate subclass for file-type data.
- ExperimentEvents(FileTypeData)
    Preset for ExperimentEvents CSV data.
- RotationData(FileTypeData)
    Preset for rotation encoder CSV data (Inner/Outer/Nosepoke).
- VideoData(FileTypeData)
    Preset for VideoData CSV data.
- VisualEnvironment(FileTypeData)
    Preset for VisualEnvironment CSV data.
- RingDebugData(FileTypeData)
    Preset for ring-debug YAML data (metadata + trials).
'''



################################################################################
# Imports
################################################################################

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from data_conduit.datasources.monosource.monosource_core import MonoSource, _build_data_arrays
from data_conduit.core.io import collect_dfs, read_csv, read_json, split_jsonl, split_yaml
from data_conduit.core.utils import starts_with

################################################################################


def _read_path(path: str | Path, **kwargs) -> Path:
    '''
    Return a Path unchanged.

    Used for split file types so collect_dfs only keeps matching files as
    Path leaves, which are then post-processed by FileTypeData.

    Parameters
    ----------
    path : str or Path
        File path to return as a Path.
    **kwargs
        Ignored. Allows this function to be used as a reader in collect_dfs
        without needing to know the file_type in advance.
    Returns
    -------
    Path
        The input path converted to a Path object.
    '''
    return Path(path)


################################################################################
# FileTypeData
################################################################################


class FileTypeData(MonoSource):
    '''
    MonoSource for common file-type data (CSV, JSON, JSONL, YAML).

    Extends MonoSource to simplify loading directories of flat files.
    Analogous to Device: maps a file_type string to the appropriate
    reader, uses device_type to filter folders via l0_selector, and
    applies post-load column/index renaming.

    Parameters
    ----------
    dfs_dict : dict or None
        Optional pre-built nested dictionary of DataFrames. If provided,
        collect_dfs is not called and this dict is used directly.
    experiment_directory_path : str or Path
        Root directory to walk.
    device_type : str or None
        Folder prefix to filter on (e.g. 'ExperimentEvents', 'VideoData').
        Maps to l0_selector=starts_with(device_type).
        If None, no folder filtering is applied.
    file_type : str
        File format to load. Determines which reader and extension to use.
        Supported: 'csv', 'json', 'jsonl', 'yml', 'yaml'.
    reader_kwargs : dict or None
        Kwargs passed to the reader function for every matching file.
        e.g. {'header': 0, 'index_col': 0} for CSV.
        If None, the reader uses its built-in defaults.
    rename_columns_dict : dict or None
        Column renaming applied to all leaf DataFrames after loading.
        e.g. {'Value': 'Event'}
    rename_index_dict : str or dict or None
        Index renaming applied to all leaf DataFrames after loading.
        If str, renames the index to that name (e.g. 'Time').
        If dict, passed to df.index.rename().
    filetype_data_arrays : dict or None
        Mapping of friendly names to level selectors for extracting
        named DataArrays from dfs_dict. Uses the same l{n}_selector
        format as collect_dfs.
        e.g. {'events': {'l0_selector': 'ExperimentEvents'}}
    keep_empty : bool
        If True, preserve empty subdirectories as empty dicts.
    flatten : bool
        If True, flatten the output dict.
    separator : str
        Separator for flattened keys.
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.
        e.g. l1_selector=['subfolder1'], l2_selector=lambda k: ...

    Attributes
    ----------
    dfs_dict : dict
        Nested dictionary of DataFrames from collect_dfs or provided directly.
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from filetype_data_arrays.
    df : pd.DataFrame or dict
        Backward-compatible convenience property. Returns the single leaf
        DataFrame from dfs_dict when there is only one file. For split
        types (JSONL/YAML), returns {'metadata': df, 'trials': df}.
        If multiple leaves exist, returns the full nested dict.
    '''

    #===========================================================================
    # File-type -> (extension, reader, split_fn) mapping
    #===========================================================================
    _FILETYPE_MAP = {
        'csv': ('.csv', read_csv, None),
        'json': ('.json', read_json, None),
        'jsonl': ('.jsonl', _read_path, split_jsonl),
        'yml': ('.yml', _read_path, split_yaml),
        'yaml': ('.yaml', _read_path, split_yaml),
    }

    def __init__(self,
                 dfs_dict: dict | None = None,
                 experiment_directory_path: str | Path | None = None,
                 device_type: str | None = None,
                 file_type: str = 'csv',
                 reader_kwargs: dict | None = None,
                 rename_columns_dict: dict | None = None,
                 rename_index_dict: str | dict | None = None,
                 filetype_data_arrays: dict | None = None,
                 keep_empty: bool = False,
                 flatten: bool = False,
                 separator: str = ':',
                 verbose: bool = False,
                 **kwargs,
                 ):
        '''
        Initialise FileTypeData.

        Resolves file_type to a reader, loads via collect_dfs, applies
        post-processing (double-folder collapse, split handling, renaming),
        then passes the clean dfs_dict to MonoSource.

        Parameters
        ----------
        dfs_dict : dict or None
            Optional pre-built nested dictionary of DataFrames. If provided,
            collect_dfs is not called and this dict is used directly.

        experiment_directory_path : str or Path or None
            Path to the experiment directory. Used to build dfs_dict if not provided.

        device_type : str or None
            Device type to filter files in the experiment directory.

        file_type : str
            One of 'csv', 'json', 'jsonl', 'yml', 'yaml'. Determines the reader to use.

        reader_kwargs : dict or None
            Additional keyword arguments passed to the reader function.

        rename_columns_dict : dict or None
            Dictionary mapping old column names to new column names.

        rename_index_dict : str or dict or None
            Dictionary mapping old index names to new index names.

        filetype_data_arrays : dict or None
            Dictionary mapping file types to DataArrays.

        keep_empty : bool
            If True, preserve empty subdirectories as empty dicts.

        flatten : bool
            If True, flatten the output dict.

        separator : str
            Separator for flattened keys.

        verbose : bool
            If True, print warnings during processing.

        **kwargs
            Additional level selectors passed to collect_dfs.
            e.g. l1_selector=['subfolder1'], l2_selector=lambda k: ...

        '''
        self.device_type = device_type
        self.file_type = file_type
        self.rename_columns_dict = rename_columns_dict
        self.rename_index_dict = rename_index_dict
        self.filetype_data_arrays = filetype_data_arrays
        self.verbose = verbose

        # Build dfs_dict if not provided
        if dfs_dict is None and experiment_directory_path is not None:
            readers, mapped_reader_kwargs, self._split_fn = self._resolve_readers(
                file_type, reader_kwargs
            )

            if device_type is not None and 'l0_selector' not in kwargs:
                kwargs['l0_selector'] = starts_with(device_type)

            dfs_dict = collect_dfs(
                base_path=experiment_directory_path,
                readers=readers,
                reader_kwargs=mapped_reader_kwargs,
                keep_empty=keep_empty,
                flatten=flatten,
                separator=separator,
                verbose=verbose,
                **kwargs,
            )

            # Post-process
            dfs_dict = self._collapse_double_folders_dict(dfs_dict)
            if self._split_fn is not None:
                dfs_dict = self._apply_split_to_leaves(dfs_dict, self._split_fn, verbose)
            dfs_dict = self._rename_leaves(dfs_dict, rename_columns_dict, rename_index_dict)
        else:
            self._split_fn = None

        # Pass clean dfs_dict to MonoSource
        super().__init__(
            dfs_dict=dfs_dict,
            experiment_directory_path=experiment_directory_path,
            monosource_data_arrays=filetype_data_arrays,
            verbose=verbose,
        )

    #===========================================================================
    # Backward-compatible .df property
    #===========================================================================

    @property
    def df(self):
        '''
        Backward-compatible access to the loaded data.

        Walks dfs_dict and returns the single leaf DataFrame (or dict
        for split types like JSONL/YAML). If multiple leaves exist,
        returns the full nested dict so nothing silently drops data.
        '''
        def _find_leaf(d):
            if not isinstance(d, dict):
                return d
            if len(d) == 1:
                return _find_leaf(next(iter(d.values())))
            return d
        return _find_leaf(self.dfs_dict)

    #===========================================================================
    # Reader resolution
    #===========================================================================

    @classmethod
    def _resolve_readers(cls,
                         file_type: str,
                         reader_kwargs: dict | None,
                         ) -> tuple[dict | None, dict | None, Callable | None]:
        '''
        Map a file_type string to readers and reader_kwargs dicts
        suitable for collect_dfs.

        Parameters
        ----------
        file_type : str
            One of 'csv', 'json', 'jsonl', 'yml', 'yaml'.
        reader_kwargs : dict or None
            Flat kwargs dict for the reader.

        Returns
        -------
        tuple[dict | None, dict | None, Callable | None]
            (readers, mapped_reader_kwargs, split_fn) for collect_dfs.
        '''
        if file_type not in cls._FILETYPE_MAP:
            raise ValueError(
                f"Unsupported file_type: '{file_type}'. "
                f"Supported: {list(cls._FILETYPE_MAP.keys())}. "
                f"For custom formats, pass readers directly to MonoSource."
            )

        extension, reader_fn, split_fn = cls._FILETYPE_MAP[file_type]
        readers = {extension: reader_fn}
        mapped_kwargs = {extension: reader_kwargs} if reader_kwargs else None
        return readers, mapped_kwargs, split_fn

    #===========================================================================
    # Post-processing helpers (all static, operate on dicts)
    #===========================================================================

    @staticmethod
    def _collapse_double_folders_dict(dfs_dict: dict) -> dict:
        '''
        Collapse double folders in dfs_dict.

        Bonsai often creates {DeviceType: {DeviceType: {files...}}}
        structures. This collapses them to {DeviceType: {files...}}.
        '''
        collapsed = {}
        for key, value in dfs_dict.items():
            if isinstance(value, dict) and list(value.keys()) == [key]:
                collapsed[key] = value[key]
            else:
                collapsed[key] = value
        return collapsed

    @staticmethod
    def _apply_split_to_leaves(d: dict,
                               split_fn: Callable,
                               verbose: bool = False,
                               ) -> dict:
        '''
        Recursively walk a nested dict and apply split_fn to Path leaves.

        Parameters
        ----------
        d : dict
            Nested dict (dfs_dict) potentially containing Path leaves.
        split_fn : Callable
            Function with signature (path) -> (metadata_df, trials_df).
        verbose : bool
            If True, print progress.

        Returns
        -------
        dict
            Same structure but Path leaves replaced with
            {'metadata': df, 'trials': df}.
        '''
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = FileTypeData._apply_split_to_leaves(
                    value, split_fn, verbose
                )
            elif isinstance(value, Path):
                try:
                    meta_df, trials_df = split_fn(value, verbose=verbose)
                    result[key] = {'metadata': meta_df, 'trials': trials_df}
                except Exception as error:
                    if verbose:
                        print(f"Warning: Split failed for '{key}' ({value}): {error}")
            else:
                result[key] = value
        return result

    @staticmethod
    def _rename_leaves(d: dict,
                       columns_dict: dict | None,
                       index_rename: str | dict | None,
                       ) -> dict:
        '''
        Recursively walk a nested dict and apply column/index renames
        to all DataFrame leaves.

        Parameters
        ----------
        d : dict
            Nested dict (dfs_dict).
        columns_dict : dict or None
            Column renaming dict passed to df.rename(columns=...).
        index_rename : str, dict, or None
            If str, renames the index name to that string.
            If dict, passed to df.rename(index=...).

        Returns
        -------
        dict
            Same structure with renamed DataFrames.
        '''
        if columns_dict is None and index_rename is None:
            return d

        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = FileTypeData._rename_leaves(
                    value,
                    columns_dict,
                    index_rename,
                )
            elif isinstance(value, pd.DataFrame):
                dataframe = value
                if columns_dict is not None:
                    dataframe = dataframe.rename(columns=columns_dict)
                if index_rename is not None:
                    if isinstance(index_rename, str):
                        dataframe = dataframe.copy()
                        dataframe.index.name = index_rename
                    elif isinstance(index_rename, dict):
                        dataframe = dataframe.rename(index=index_rename)
                result[key] = dataframe
            else:
                result[key] = value
        return result


################################################################################
# ExperimentEvents
################################################################################


class ExperimentEvents(FileTypeData):
    '''
    Preset config for ExperimentEvents CSV data.

    Loads CSV files from ExperimentEvents subfolder(s), renames the
    'Value' column to 'Event', and ensures the index is named 'Time'.

    Parameters
    ----------
    experiment_directory_path : str or Path or None
        Path to the experiment directory.
    device_type : str
        Folder prefix. Default 'ExperimentEvents'.
    reader_kwargs : dict or None
        Custom kwargs for read_csv.
    rename_columns_dict : dict
        Column renaming. Default {'Value': 'Event'}.
    rename_index_dict : str
        Index name. Default 'Time'.
    filetype_data_arrays : dict
        Mapping of friendly names to level selectors.
        Default: {'events': {'l0_selector': 'ExperimentEvents'}}
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.

    Attributes
    ----------
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from filetype_data_arrays.
        Default ExperimentEvents data_arrays:
        {
            'events': xr.DataArray for ExperimentEvents CSV data,
        }
    df : pd.DataFrame
        Backward-compatible access to the single loaded DataFrame.
    '''

    def __init__(self,
                 experiment_directory_path: str | Path | None = None,
                 device_type: str = 'ExperimentEvents',
                 reader_kwargs: dict | None = None,
                 rename_columns_dict: dict | None = None,
                 rename_index_dict: str | dict | None = 'Time',
                 filetype_data_arrays: dict | None = None,
                 verbose: bool = False,
                 **kwargs,
                 ):
        '''Initialise ExperimentEvents with CSV preset.'''
        if rename_columns_dict is None:
            rename_columns_dict = {'Value': 'Event'}
        if filetype_data_arrays is None:
            filetype_data_arrays = {'events': {'l0_selector': 'ExperimentEvents'}}

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            device_type=device_type,
            file_type='csv',
            reader_kwargs=reader_kwargs,
            rename_columns_dict=rename_columns_dict,
            rename_index_dict=rename_index_dict,
            filetype_data_arrays=filetype_data_arrays,
            verbose=verbose,
            **kwargs,
        )


################################################################################
# RotationData
################################################################################


class RotationData(FileTypeData):
    '''
    Preset config for rotation encoder CSV data.

    Loads CSV files from a rotation device subfolder (InnerRotation,
    OuterRotation, or NosepokeRotation), renames 'Value' to 'Rotation',
    and optionally applies angular unit conversion and range wrapping.

    Parameters
    ----------
    experiment_directory_path : str or Path or None
        Path to the experiment directory.
    device_type : str or None
        Folder prefix. Typically one of 'InnerRotation',
        'OuterRotation', or 'NosepokeRotation'.
    reader_kwargs : dict or None
        Custom kwargs for read_csv.
    rename_columns_dict : dict
        Column renaming. Default {'Value': 'Rotation'}.
    rename_index_dict : str
        Index name. Default 'Time'.
    angular_unit_conversion : str or None
        Unit conversion to apply: 'deg2rad' or 'rad2deg'.
        If None, no conversion is applied.
    angular_range : list[float] or None
        Two-element list [min, max] to wrap angular values within.
        e.g. [0, 360], [-180, 180], [0, 2*pi].
        If None, no wrapping is applied.
    filetype_data_arrays : dict or None
        Mapping of friendly names to level selectors.
        No default — device_type varies per instantiation.
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.

    Attributes
    ----------
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from filetype_data_arrays (if provided).
    df : pd.DataFrame
        Backward-compatible access to the single loaded DataFrame.
    '''

    def __init__(self,
                 experiment_directory_path: str | Path | None = None,
                 device_type: str | None = None,
                 reader_kwargs: dict | None = None,
                 rename_columns_dict: dict | None = None,
                 rename_index_dict: str | dict | None = 'Time',
                 angular_unit_conversion: str | None = None,
                 angular_range: list | None = None,
                 filetype_data_arrays: dict | None = None,
                 verbose: bool = False,
                 **kwargs,
                 ):
        '''Initialise RotationData with CSV preset and angular processing.'''
        if rename_columns_dict is None:
            rename_columns_dict = {'Value': 'Rotation'}

        self.angular_unit_conversion = angular_unit_conversion
        self.angular_range = angular_range

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            device_type=device_type,
            file_type='csv',
            reader_kwargs=reader_kwargs,
            rename_columns_dict=rename_columns_dict,
            rename_index_dict=rename_index_dict,
            filetype_data_arrays=filetype_data_arrays,
            verbose=verbose,
            **kwargs,
        )

        # Apply angular transforms after all base post-processing
        if self.angular_unit_conversion is not None or self.angular_range is not None:
            self._apply_angular_transforms()
            # Rebuild data_arrays since dfs_dict changed
            if self.filetype_data_arrays is not None:
                self.data_arrays = _build_data_arrays(
                    dfs_dict=self.dfs_dict,
                    monosource_data_arrays=self.filetype_data_arrays,
                    verbose=self.verbose,
                )

    def _apply_angular_transforms(self):
        '''
        Apply angular unit conversion and range wrapping to all
        Rotation columns in leaf DataFrames of dfs_dict.
        '''
        self.dfs_dict = self._transform_rotation_leaves(
            self.dfs_dict,
            self.angular_unit_conversion,
            self.angular_range,
            self.verbose,
        )

    @staticmethod
    def _transform_rotation_leaves(d: dict,
                                   conversion: str | None,
                                   angular_range: list | None,
                                   verbose: bool,
                                   ) -> dict:
        '''
        Recursively apply angular transforms to Rotation columns in
        all DataFrame leaves.
        '''
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = RotationData._transform_rotation_leaves(
                    value, conversion, angular_range, verbose
                )
            elif isinstance(value, pd.DataFrame) and 'Rotation' in value.columns:
                dataframe = value.copy()
                dataframe['Rotation'] = pd.to_numeric(dataframe['Rotation'], errors='coerce')

                if dataframe['Rotation'].isna().any():
                    bad_rows = int(dataframe['Rotation'].isna().sum())
                    raise ValueError(
                        f"RotationData: 'Rotation' column in '{key}' contains "
                        f"{bad_rows} non-numeric value(s); cannot apply "
                        f"angular transforms."
                    )

                if conversion == 'deg2rad':
                    dataframe['Rotation'] = np.deg2rad(dataframe['Rotation'])
                elif conversion == 'rad2deg':
                    dataframe['Rotation'] = np.rad2deg(dataframe['Rotation'])
                elif conversion is not None:
                    raise ValueError(
                        f"Invalid angular_unit_conversion: '{conversion}'. "
                        f"Must be 'deg2rad' or 'rad2deg'."
                    )

                if angular_range is not None:
                    if not (isinstance(angular_range, list) and len(angular_range) == 2):
                        raise ValueError(
                            f"angular_range must be a list of two floats "
                            f"[min, max]. Got: {angular_range}"
                        )
                    min_value, max_value = angular_range
                    dataframe['Rotation'] = (
                        (dataframe['Rotation'] - min_value) % (max_value - min_value)
                    ) + min_value

                if verbose:
                    if conversion:
                        print(f"Applied {conversion} to '{key}'")
                    if angular_range:
                        print(f"Wrapped '{key}' within {angular_range}")

                result[key] = dataframe
            else:
                result[key] = value
        return result


################################################################################
# VideoData
################################################################################


class VideoData(FileTypeData):
    '''
    Preset config for VideoData CSV data.

    Loads CSV files from VideoData subfolder(s), renames chunk data
    columns to friendly names (FrameID, Timestamp).

    Parameters
    ----------
    experiment_directory_path : str or Path or None
        Path to the experiment directory.
    device_type : str
        Folder prefix. Default 'VideoData'.
    reader_kwargs : dict or None
        Custom kwargs for read_csv.
    rename_columns_dict : dict
        Column renaming. Default maps ChunkData fields.
    rename_index_dict : str
        Index name. Default 'Time'.
    filetype_data_arrays : dict
        Mapping of friendly names to level selectors.
        Default: {'video': {'l0_selector': 'VideoData'}}
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.

    Attributes
    ----------
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from filetype_data_arrays.
        Default VideoData data_arrays:
        {
            'video': xr.DataArray for VideoData CSV data,
        }
    df : pd.DataFrame
        Backward-compatible access to the single loaded DataFrame.
    '''

    def __init__(self,
                 experiment_directory_path: str | Path | None = None,
                 device_type: str = 'VideoData',
                 reader_kwargs: dict | None = None,
                 rename_columns_dict: dict | None = None,
                 rename_index_dict: str | dict | None = 'Time',
                 filetype_data_arrays: dict | None = None,
                 verbose: bool = False,
                 **kwargs,
                 ):
        '''Initialise VideoData with CSV preset.'''
        if rename_columns_dict is None:
            rename_columns_dict = {'Value.ChunkData.FrameID': 'FrameID', 'Value.ChunkData.Timestamp': 'Timestamp'}
        if filetype_data_arrays is None:
            filetype_data_arrays = {'video': {'l0_selector': 'VideoData'}}

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            device_type=device_type,
            file_type='csv',
            reader_kwargs=reader_kwargs,
            rename_columns_dict=rename_columns_dict,
            rename_index_dict=rename_index_dict,
            filetype_data_arrays=filetype_data_arrays,
            verbose=verbose,
            **kwargs,
        )


################################################################################
# VisualEnvironment
################################################################################


class VisualEnvironment(FileTypeData):
    '''
    Preset config for VisualEnvironment CSV data.

    Loads CSV files from VisualEnvironment subfolder(s) using custom
    column names for the headerless Bonsai output format.

    Parameters
    ----------
    experiment_directory_path : str or Path or None
        Path to the experiment directory.
    device_type : str
        Folder prefix. Default 'VisualEnvironment'.
    reader_kwargs : dict
        Custom kwargs for read_csv. Defaults provide column names
        for the standard VisualEnvironment format.
    rename_columns_dict : dict or None
        Column renaming. Default None (names set via reader_kwargs).
    rename_index_dict : str
        Index name. Default 'Time'.
    filetype_data_arrays : dict
        Mapping of friendly names to level selectors.
        Default: {'visual_environment': {'l0_selector': 'VisualEnvironment'}}
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.

    Attributes
    ----------
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from filetype_data_arrays.
        Default VisualEnvironment data_arrays:
        {
            'visual_environment': xr.DataArray for VisualEnvironment CSV data,
        }
    df : pd.DataFrame
        Backward-compatible access to the single loaded DataFrame.
    '''

    def __init__(self,
                 experiment_directory_path: str | Path | None = None,
                 device_type: str = 'VisualEnvironment',
                 reader_kwargs: dict | None = None,
                 rename_columns_dict: dict | None = None,
                 rename_index_dict: str | dict | None = 'Time',
                 filetype_data_arrays: dict | None = None,
                 verbose: bool = False,
                 **kwargs,
                 ):
        '''Initialise VisualEnvironment with CSV preset and custom column names.'''
        if reader_kwargs is None:
            reader_kwargs = {
                'names': ['Time', 'Value', 'Landmark', 'Landmark_Proximal', 'Gratings', 'Firefly'],
                'skiprows': 1,
                'header': None,
                'engine': 'python',
                'index_col': 0,
            }
        if filetype_data_arrays is None:
            filetype_data_arrays = {'visual_environment': {'l0_selector': 'VisualEnvironment'}}

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            device_type=device_type,
            file_type='csv',
            reader_kwargs=reader_kwargs,
            rename_columns_dict=rename_columns_dict,
            rename_index_dict=rename_index_dict,
            filetype_data_arrays=filetype_data_arrays,
            verbose=verbose,
            **kwargs,
        )


################################################################################
# RingDebugData
################################################################################


class RingDebugData(FileTypeData):
    '''
    Preset config for ring-debug YAML data (metadata + trials).

    Loads YAML files from the experiment directory (ring-debug.yml
    typically sits at the root, not in a device subfolder). Each YAML
    file is split into metadata and trials DataFrames stored as a
    nested dict: {stem: {'metadata': df, 'trials': df}}.

    Parameters
    ----------
    experiment_directory_path : str or Path or None
        Path to the experiment directory.
    device_type : str
        Folder prefix. Default 'ring-debug'.
    rename_columns_dict : dict or None
        Column renaming. Default None.
    rename_index_dict : str
        Index name. Default 'Time'.
    filetype_data_arrays : dict
        Mapping of friendly names to level selectors.
        Default: {'ring_debug': {'l0_selector': 'ring-debug'}}
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.

    Attributes
    ----------
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from filetype_data_arrays.
        Default RingDebugData data_arrays:
        {
            'ring_debug': xr.DataArray for ring-debug YAML data,
        }
    df : dict
        Backward-compatible access. Returns {'metadata': df, 'trials': df}.
    '''

    def __init__(self,
                 experiment_directory_path: str | Path | None = None,
                 device_type: str = 'ring-debug',
                 rename_columns_dict: dict | None = None,
                 rename_index_dict: str | dict | None = 'Time',
                 filetype_data_arrays: dict | None = None,
                 verbose: bool = False,
                 **kwargs,
                 ):
        '''Initialise RingDebugData with YAML split preset.'''
        if filetype_data_arrays is None:
            filetype_data_arrays = {'ring_debug': {'l0_selector': 'ring-debug'}}
        super().__init__(
            experiment_directory_path=experiment_directory_path,
            device_type=device_type,
            file_type='yml',
            rename_columns_dict=rename_columns_dict,
            rename_index_dict=rename_index_dict,
            filetype_data_arrays=filetype_data_arrays,
            verbose=verbose,
            **kwargs,
        )


################################################################################
# SessionSettings
################################################################################


class SessionSettings(FileTypeData):
    '''
    Preset config for SessionSettings JSONL data (metadata + per-trial settings).

    Loads JSONL files from the SessionSettings subfolder. Each JSONL record has the 
    shape {"seconds": ..., "value": {"metadata": {...}, "trials": [...]}}. The `split_jsonl`
    function produces a {'metadata': df, 'trials': df} pair per file. 

    Trial-level list columns (e.g. `timeToTarget`, `targetZone.zoneRange`) are expanded into 
    suffixed scalar columns (`timeToTarget_0`, `timeToTarget_1`, ..., `targetZone.zoneRange_0`, etc.),
    so analysis code can index by column name.
    A leading ``trial`` column is added to the trials DataFrame with labels
    ``trial_1``, ``trial_2``, etc. so displayed and exported tables clearly
    identify each row.

    Schema tolerance (filling missing columns with defaults, retaining or dropping unknown columns) is 
    controlled by the `expected_columns` and `strict` parameters. Missing expected columns are filled 
    with the supplied default value; unknown columns are retained by default and dropped when `strict=True`.


    Parameters
    ----------
    experiment_directory_path : str or Path or None
        Path to the experiment directory.
    device_type : str or None
        Folder prefix. Default 'SessionSettings'. Set to None to search
        the directory root (e.g. when the JSONL sits alongside other files).
    expected_columns : dict or None
        Mapping of {column_name: default_value} for expected trial columns.
        Missing columns are added filled with the default value.
        If None, uses SessionSettings.DEFAULT_EXPECTED_COLUMNS.
    strict : bool
        If True, restrict the trials frame to exactly the keys of
        ``expected_columns`` (unknown columns are dropped).
        If False (default), unknown columns are retained.
    rename_columns_dict : dict or None
        Column renaming. Default None.
    rename_index_dict : str or dict or None
        Index renaming. Default None.
    filetype_data_arrays : dict
        Mapping of friendly names to level selectors.
        Default: {'session_settings': {'l0_selector': 'SessionSettings'}}
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.

    Attributes
    ----------
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from filetype_data_arrays.
    df : dict
        Backward-compatible access. Returns {'metadata': df, 'trials': df}.

    DEFAULT_EXPECTED_COLUMNS : dict
        Default expected trial columns (all default to ``None``):

        ========================== =========================================
        Group                      Columns
        ========================== =========================================
        Runtime                    ``maxRuntime``
        Time-to-target             ``timeToTarget_0``, ``_1``
        Target zone                ``targetZone.zoneRange_0`` … ``_3``
        Landmark                   ``landmark.draw``, ``.cue``, ``.angleOffset_0``, ``_1``
        Landmark proximal          ``landmarkProximal.draw``
        Grating                    ``grating.draw``, ``.spatialFrequency``, ``.temporalFrequency``, ``.contrast``
        Firefly                    ``firefly.draw``, ``.dotCount``, ``.coherence``, ``.speed``
        Arena rotation             ``arenaRotation.rotationRanges_0`` … ``_3``
        Nosepoke rotation          ``nosepokeRotation.rotationRanges_0``, ``_1``
        Mismatch rotation          ``mismatchRotation.rotationRanges_0``, ``_1``
        Reward tone                ``rewardTone.frequency_0``, ``_1``, ``.attenuation_0``, ``_1``
        Reward                     ``rewardTimeout``, ``rewardPortIndicators``
        Masking sound              ``maskingSound.play``
        Visual                     ``overlayAlpha``, ``visualRevealDelay_0``, ``_1``
        ========================== =========================================

    '''

    DEFAULT_EXPECTED_COLUMNS: dict = {
        'maxRuntime': None,
        'timeToTarget_0': None,
        'timeToTarget_1': None,
        'targetZone.zoneRange_0': None,
        'targetZone.zoneRange_1': None,
        'targetZone.zoneRange_2': None,
        'targetZone.zoneRange_3': None,
        'landmark.draw': None,
        'landmark.cue': None,
        'landmark.angleOffset_0': None,
        'landmark.angleOffset_1': None,
        'landmarkProximal.draw': None,
        'grating.draw': None,
        'grating.spatialFrequency': None,
        'grating.temporalFrequency': None,
        'grating.contrast': None,
        'firefly.draw': None,
        'firefly.dotCount': None,
        'firefly.coherence': None,
        'firefly.speed': None,
        'arenaRotation.rotationRanges_0': None,
        'arenaRotation.rotationRanges_1': None,
        'arenaRotation.rotationRanges_2': None,
        'arenaRotation.rotationRanges_3': None,
        'nosepokeRotation.rotationRanges_0': None,
        'nosepokeRotation.rotationRanges_1': None,
        'mismatchRotation.rotationRanges_0': None,
        'mismatchRotation.rotationRanges_1': None,
        'rewardTone.frequency_0': None,
        'rewardTone.frequency_1': None,
        'rewardTone.attenuation_0': None,
        'rewardTone.attenuation_1': None,
        'rewardTimeout': None,
        'rewardPortIndicators': None,
        'maskingSound.play': None,
        'overlayAlpha': None,
        'visualRevealDelay_0': None,
        'visualRevealDelay_1': None,
    }

    def __init__(self,
                 experiment_directory_path: str | Path | None = None,
                 device_type: str | None = 'SessionSettings',
                 expected_columns: dict | None = None,
                 strict: bool = False,
                 rename_columns_dict: dict | None = None,
                 rename_index_dict: str | dict | None = None,
                 filetype_data_arrays: dict | None = None,
                 verbose: bool = False,
                 **kwargs,
                 ):
        '''
        Initialise SessionSettings with JSONL split preset.
        
        Parameters
        ----------
        experiment_directory_path : str or Path or None
            Path to the experiment directory. SessionSettings JSONL files typically sit in a 'SessionSettings' subfolder, 
            but setting device_type to None allows searching the root directory (e.g. when the JSONL sits alongside other files).
            If None, dfs_dict must be provided directly.
        device_type : str or None
            Folder prefix. Default 'SessionSettings'. Set to None to search the directory root.
        expected_columns : dict or None
            Mapping of {column_name: default_value} for expected trial columns. Missing columns are added filled with the default value. 
            If None, uses SessionSettings.DEFAULT_EXPECTED_COLUMNS.
        strict : bool
            If True, only the expected columns are kept in the DataFrame. Extra columns are dropped. If False (default), extra columns are retained.
        rename_columns_dict : dict or None
            Column renaming. Default None.
        rename_index_dict : str or dict or None
            Index renaming. Default None.
        filetype_data_arrays : dict
            Mapping of friendly names to level selectors.
            Default: {'session_settings': {'l0_selector': 'SessionSettings'}}
        verbose : bool
            If True, print warnings during processing.
        **kwargs
            Additional level selectors passed to collect_dfs.
        
        Attributes
        ----------
        data_arrays : dict[str, xr.DataArray]
            Named DataArrays built from filetype_data_arrays. Default:
            {'session_settings': xr.DataArray for SessionSettings JSONL data}
        df : dict
            Backward-compatible access. Returns {'metadata': df, 'trials': df}.
        DEFAULT_EXPECTED_COLUMNS : dict
            Default expected columns and their default values for trial-level data. See class docstring for details.

        '''
        if filetype_data_arrays is None:
            filetype_data_arrays = {'session_settings': {'l0_selector': 'SessionSettings'}}
        self.expected_columns = (
            dict(self.DEFAULT_EXPECTED_COLUMNS)
            if expected_columns is None
            else expected_columns
        )
        self.strict = strict

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            device_type=device_type,
            file_type='jsonl',
            rename_columns_dict=rename_columns_dict,
            rename_index_dict=rename_index_dict,
            filetype_data_arrays=filetype_data_arrays,
            verbose=verbose,
            **kwargs,
        )

        self.dfs_dict = self._reconcile_trial_schemas(
            self.dfs_dict, self.expected_columns, self.strict, self.verbose,
        )

        if self.filetype_data_arrays is not None:
            self.data_arrays = _build_data_arrays(
                dfs_dict=self.dfs_dict,
                monosource_data_arrays=self.filetype_data_arrays,
                verbose=self.verbose,
            )

    @staticmethod
    def _reconcile_trial_schemas(d: dict,
                                 expected: dict,
                                 strict: bool,
                                 verbose: bool,
                                 ) -> dict:
        '''
        Walk dfs_dict and apply list-expansion + schema reconciliation to any
        {'metadata': df, 'trials': df} leaf produced by split_jsonl.

        Parameters
        ----------
        d : dict
            Nested dict (dfs_dict) potentially containing {'metadata': df, 'trials': df} leaves.
        expected : dict
            Mapping of {column_name: default_value} for expected trial columns. Missing columns are added filled with the default value.
        strict : bool
            If True, only the expected columns are kept in the DataFrame. Extra columns are dropped. If False, extra columns are retained.
        verbose : bool
            If True, print warnings about missing or extra columns during reconciliation.
        
        Returns
        -------
        dict
            Nested dict with reconciled trial schemas.
        
            
        '''
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                if (
                    'trials' in value
                    and isinstance(value['trials'], pd.DataFrame)
                ):
                    reconciled = dict(value)
                    reconciled['trials'] = SessionSettings._reconcile_trials_frame(
                        value['trials'], expected, strict, verbose, key,
                    )
                    result[key] = reconciled
                else:
                    result[key] = SessionSettings._reconcile_trial_schemas(
                        value, expected, strict, verbose,
                    )
            else:
                result[key] = value
        return result

    @staticmethod
    def _reconcile_trials_frame(df: pd.DataFrame,
                                expected: dict,
                                strict: bool,
                                verbose: bool,
                                source_key: str,
                                ) -> pd.DataFrame:
        '''
        Expand list columns, add missing expected columns, optionally drop extras.
        
        Parameters
        ----------
        df : pd.DataFrame
            The trials DataFrame to reconcile. Expected to have one row per trial, but may contain list-valued columns.
        expected : dict
            Mapping of {column_name: default_value} for expected trial columns. Missing columns are added filled with the default value.
        strict : bool
            If True, only the expected columns are kept in the DataFrame. Extra columns are dropped. If False, extra columns are retained.
        verbose : bool
            If True, print warnings about missing or extra columns.
        source_key : str
            Identifier for the source of this DataFrame, used in warning messages.

        Returns
        -------
        pd.DataFrame
            The reconciled DataFrame with expanded list columns, all expected columns present, and extras retained or dropped according to `strict`.
        

        '''
        df = SessionSettings._expand_list_columns(df)

        missing = [c for c in expected if c not in df.columns]
        extras = [c for c in df.columns if c not in expected]

        for col in missing:
            df[col] = expected[col]

        if strict:
            df = df.loc[:, list(expected.keys())]

        df = SessionSettings._add_trial_labels(df)

        if verbose:
            if missing:
                print(f"SessionSettings '{source_key}': filled missing columns {missing}")
            if extras:
                action = 'dropped' if strict else 'retained'
                print(f"SessionSettings '{source_key}': {action} extra columns {extras}")

        return df

    @staticmethod
    def _add_trial_labels(df: pd.DataFrame) -> pd.DataFrame:
        '''
        Add a first-column trial label for display and CSV export.

        The raw JSONL trials naturally arrive as one row per trial, but pandas
        gives them a plain RangeIndex. A visible column survives notebook
        display and ``to_csv(index=False)`` better than relying on that index.
        '''
        if 'trial' in df.columns:
            return df

        labelled = df.copy()
        labelled.insert(
            0,
            'trial',
            [f'trial_{trial_number}' for trial_number in range(1, len(labelled) + 1)],
        )
        return labelled

    @staticmethod
    def _expand_list_columns(df: pd.DataFrame) -> pd.DataFrame:
        '''
        Expand any list-valued column into suffixed scalar columns (``col_0``, ``col_1``, ...).
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to expand.

        Returns
        -------
        pd.DataFrame
            The DataFrame with list-valued columns expanded into suffixed scalar columns.
        '''
        new_df = df.copy()
        for col in df.columns:
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            if not non_null.apply(lambda v: isinstance(v, list)).any():
                continue
            max_len = int(
                df[col].apply(lambda v: len(v) if isinstance(v, list) else 0).max()
            )
            for i in range(max_len):
                new_df[f"{col}_{i}"] = df[col].apply(
                    lambda v, i=i: v[i] if isinstance(v, list) and i < len(v) else None
                )
            new_df = new_df.drop(columns=[col])
        return new_df


__all__ = [
    'FileTypeData',
    'ExperimentEvents',
    'RotationData',
    'VideoData',
    'VisualEnvironment',
    'RingDebugData',
    'SessionSettings',
]

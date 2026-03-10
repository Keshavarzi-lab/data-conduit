'''
FileTypeData and preset subclasses.
--------------------------------

Description:
    FileTypeData extends DataSource for common file-type data (CSV, JSON,
    JSONL, YAML). Analogous to Device but for flat-file formats.
    Handles device_type folder filtering, reader resolution from file_type
    strings, and post-load column/index renaming.

    Subclasses provide preset configurations for specific data sources
    (ExperimentEvents, RotationData, VideoData, etc.), each setting
    defaults for device_type, file_type, renaming, and validation.

Contents:
--------------------------------
- FileTypeData(DataSource)
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

from data_conduit.datasource.datasource_core import DataSource, _build_data_arrays
from data_conduit.io import read_csv, read_json, split_jsonl, split_yaml
from data_conduit.utils import starts_with

################################################################################


def _read_path(path: str | Path, **kwargs) -> Path:
    '''
    Return a Path unchanged.

    Used for split file types so collect_dfs only keeps matching files as
    Path leaves, which are then post-processed by FileTypeData.
    '''
    return Path(path)


################################################################################
# FileTypeData
################################################################################


class FileTypeData(DataSource):
    '''
    DataSource for common file-type data (CSV, JSON, JSONL, YAML).

    Extends DataSource to simplify loading directories of flat files.
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
        Mapping of friendly names to paths in dfs_dict.
        e.g. {'trials': ('ExperimentEvents', 'events')}
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

        Resolves file_type to a reader, maps device_type to an l0_selector,
        then delegates to DataSource for directory walking and file reading.
        Applies post-processing (double-folder collapse, renaming, split
        handling) after loading.
        '''
        self.device_type = device_type
        self.file_type = file_type
        self.rename_columns_dict = rename_columns_dict
        self.rename_index_dict = rename_index_dict
        self.filetype_data_arrays = filetype_data_arrays
        self.verbose = verbose

        #=== i| Resolve file_type -> reader
        readers, mapped_reader_kwargs, split_fn = self._resolve_readers(
            file_type, reader_kwargs
        )
        self._split_fn = split_fn

        #=== ii| Map device_type -> l0_selector
        if device_type is not None and 'l0_selector' not in kwargs:
            kwargs['l0_selector'] = starts_with(device_type)

        #=== iii| Load via DataSource (defer filetype_data_arrays until after post-processing)
        super().__init__(
            dfs_dict=dfs_dict,
            experiment_directory_path=experiment_directory_path,
            readers=readers,
            reader_kwargs=mapped_reader_kwargs,
            datasource_data_arrays=None,
            keep_empty=keep_empty,
            flatten=flatten,
            separator=separator,
            verbose=verbose,
            **kwargs,
        )

        #=== iv| Post-process
        self._collapse_double_folders()
        self._handle_split_types()
        self._apply_renames()

        #=== v| Build named DataArrays now that dfs_dict is in final shape
        if filetype_data_arrays is not None:
            self.data_arrays = _build_data_arrays(
                dfs_dict=self.dfs_dict,
                datasource_data_arrays=filetype_data_arrays,
                verbose=verbose,
            )

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
                f"For custom formats, pass readers directly to DataSource."
            )

        extension, reader_fn, split_fn = cls._FILETYPE_MAP[file_type]
        readers = {extension: reader_fn}
        mapped_kwargs = {extension: reader_kwargs} if reader_kwargs else None
        return readers, mapped_kwargs, split_fn

    def _collapse_double_folders(self):
        '''
        Collapse double folders in dfs_dict.

        Bonsai often creates {DeviceType: {DeviceType: {files...}}}
        structures. This collapses them to {DeviceType: {files...}}.
        Same logic as Device.
        '''
        for key in list(self.dfs_dict.keys()):
            value = self.dfs_dict[key]
            if isinstance(value, dict) and list(value.keys()) == [key]:
                self.dfs_dict[key] = value[key]

    def _handle_split_types(self):
        '''
        Handle split file types (JSONL/YAML -> metadata+trials).

        For jsonl/yml/yaml file_types, leaf values in dfs_dict are Paths
        (since a path-preserving reader was passed to collect_dfs). This
        method walks dfs_dict, finds Path leaves, and replaces them with
        {'metadata': df, 'trials': df} dicts produced by split_jsonl or
        split_yaml.
        '''
        if self._split_fn is None:
            return

        self.dfs_dict = self._apply_split_to_leaves(
            self.dfs_dict,
            self._split_fn,
            self.verbose,
        )

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

    def _apply_renames(self):
        '''
        Apply rename_columns_dict and rename_index_dict to all leaf
        DataFrames in dfs_dict.
        '''
        if self.rename_columns_dict is None and self.rename_index_dict is None:
            return

        self.dfs_dict = self._rename_leaves(
            self.dfs_dict,
            self.rename_columns_dict,
            self.rename_index_dict,
        )

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
    experiment_directory_path : str or Path
        Path to the experiment directory.
    device_type : str
        Folder prefix. Default 'ExperimentEvents'.
    reader_kwargs : dict or None
        Custom kwargs for read_csv.
    rename_columns_dict : dict
        Column renaming. Default {'Value': 'Event'}.
    rename_index_dict : str
        Index name. Default 'Time'.
    filetype_data_arrays : dict or None
        Mapping of friendly names to paths in dfs_dict.
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.
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
    experiment_directory_path : str or Path
        Path to the experiment directory.
    device_type : str or None
        Folder prefix. Must be one of 'InnerRotation',
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
        Mapping of friendly names to paths in dfs_dict.
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.
    '''

    _VALID_DEVICE_TYPES = [
        'InnerRotation', 'OuterRotation', 'NosepokeRotation',
    ]

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
        if device_type is not None and device_type not in self._VALID_DEVICE_TYPES:
            raise ValueError(
                f"RotationData device_type must be one of "
                f"{self._VALID_DEVICE_TYPES}. Got: '{device_type}'."
            )

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

        #=== Apply angular transforms after all base post-processing
        if self.angular_unit_conversion is not None or self.angular_range is not None:
            self._apply_angular_transforms()
            if self.filetype_data_arrays is not None:
                self.data_arrays = _build_data_arrays(
                    dfs_dict=self.dfs_dict,
                    datasource_data_arrays=self.filetype_data_arrays,
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
    experiment_directory_path : str or Path
        Path to the experiment directory.
    device_type : str
        Folder prefix. Default 'VideoData'.
    reader_kwargs : dict or None
        Custom kwargs for read_csv.
    rename_columns_dict : dict
        Column renaming. Default maps ChunkData fields.
    rename_index_dict : str
        Index name. Default 'Time'.
    filetype_data_arrays : dict or None
        Mapping of friendly names to paths in dfs_dict.
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.
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
            rename_columns_dict = {
                'Value.ChunkData.FrameID': 'FrameID',
                'Value.ChunkData.Timestamp': 'Timestamp',
            }

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
    experiment_directory_path : str or Path
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
    filetype_data_arrays : dict or None
        Mapping of friendly names to paths in dfs_dict.
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.
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
                'names': [
                    'Time', 'Value', 'Landmark', 'Landmark_Proximal',
                    'Gratings', 'Firefly',
                ],
                'skiprows': 1,
                'header': None,
                'engine': 'python',
                'index_col': 0,
            }

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
    experiment_directory_path : str or Path
        Path to the experiment directory.
    device_type : str
        Folder prefix. Default 'ring-debug'.
    rename_columns_dict : dict or None
        Column renaming. Default None.
    rename_index_dict : str
        Index name. Default 'Time'.
    filetype_data_arrays : dict or None
        Mapping of friendly names to paths in dfs_dict.
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.
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


__all__ = [
    'FileTypeData',
    'ExperimentEvents',
    'RotationData',
    'VideoData',
    'VisualEnvironment',
    'RingDebugData',
]

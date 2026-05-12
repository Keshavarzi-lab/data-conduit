'''
Small compatibility layer for the parts of data-conduit used by this workflow.

The analysis notebooks only need one small slice of data-conduit at runtime:
``ExperimentEvents(...).df``.  The real data-conduit package can still be used
when it is installed, but this module provides a local fallback so
``Q_C_Analysis_Workflow`` can be shared as a standalone code package.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


try:
    from data_conduit.datasources.monosource import ExperimentEvents as _ExperimentEvents
except ModuleNotFoundError:
    _ExperimentEvents = None


def _flatten_nested_dict(value: Any, parent_key: str = '', separator: str = ':') -> dict[str, Any]:
    '''
    Flatten a nested dictionary.

    This mirrors the small utility behavior needed by ``utils.trial_table``
    without requiring the full data-conduit package.
    '''
    if not isinstance(value, dict):
        return {parent_key: value} if parent_key else {'': value}

    flattened = {}
    for key, child in value.items():
        flat_key = f'{parent_key}{separator}{key}' if parent_key else str(key)
        if isinstance(child, dict):
            flattened.update(_flatten_nested_dict(child, flat_key, separator))
        else:
            flattened[flat_key] = child

    return flattened


def _read_experiment_events_csv(path: Path, reader_kwargs: dict | None) -> pd.DataFrame:
    '''
    Read one ExperimentEvents CSV using the data-conduit defaults.

    The regex separator handles event strings that contain commas inside
    braces, for example ``Poke: {Success=True, ChosenPort=16, ...}``.
    '''
    if reader_kwargs is None:
        reader_kwargs = {
            'header': 0,
            'dtype': str,
            'engine': 'python',
            'sep': r',(?![^{]*})',
            'index_col': 0,
        }

    return pd.read_csv(path, **reader_kwargs)


class StandaloneExperimentEvents:
    '''
    Minimal local replacement for ``data_conduit.ExperimentEvents``.

    It intentionally implements only the behavior used by this workflow:

    - find CSV files under an ``ExperimentEvents`` folder,
    - read them into pandas DataFrames,
    - rename ``Value`` to ``Event``,
    - name the index ``Time``,
    - expose the loaded event log through ``.df``.

    It does not implement data-conduit's DataArray support or the broader
    MonoSource API. If downstream code needs those features, install the real
    data-conduit package instead.
    '''

    def __init__(
            self,
            experiment_directory_path: str | Path | None = None,
            device_type: str = 'ExperimentEvents',
            reader_kwargs: dict | None = None,
            rename_columns_dict: dict | None = None,
            rename_index_dict: str | dict | None = 'Time',
            verbose: bool = False,
            concat_files: bool = True,
            **_: Any,
    ):
        if experiment_directory_path is None:
            raise ValueError('experiment_directory_path is required.')

        self.experiment_directory_path = Path(experiment_directory_path)
        self.device_type = device_type
        self.reader_kwargs = reader_kwargs
        self.rename_columns_dict = (
            {'Value': 'Event'} if rename_columns_dict is None else rename_columns_dict
        )
        self.rename_index_dict = rename_index_dict
        self.verbose = verbose
        self.concat_files = concat_files
        self.data_arrays = {}

        self.dfs_dict = self._load_events()

    def _event_csv_paths(self) -> list[Path]:
        '''
        Return matching ExperimentEvents CSV files for the session directory.
        '''
        root = self.experiment_directory_path
        if not root.exists():
            raise FileNotFoundError(f'Experiment directory does not exist: {root}')
        if not root.is_dir():
            raise NotADirectoryError(f'Experiment path is not a directory: {root}')

        if root.name.startswith(self.device_type):
            event_dirs = [root]
        else:
            event_dirs = [
                path for path in root.rglob('*')
                if path.is_dir() and path.name.startswith(self.device_type)
            ]

        csv_paths = []
        for event_dir in event_dirs:
            csv_paths.extend(
                path for path in event_dir.rglob('*.csv')
                if path.is_file() and path.suffix == '.csv'
            )

        csv_paths = sorted(set(csv_paths))
        if not csv_paths:
            raise FileNotFoundError(
                f'No {self.device_type} CSV files found under {root}'
            )

        return csv_paths

    def _load_events(self) -> dict:
        '''
        Load all matching event CSVs into a small nested dictionary.
        '''
        loaded = {}
        for csv_path in self._event_csv_paths():
            if self.verbose:
                print(f'[ExperimentEvents shim] {csv_path}')

            frame = _read_experiment_events_csv(csv_path, self.reader_kwargs)

            if self.rename_columns_dict:
                frame = frame.rename(columns=self.rename_columns_dict)

            if self.rename_index_dict is not None:
                frame.index = frame.index.rename(self.rename_index_dict)

            loaded[csv_path.stem] = frame

        return {self.device_type: loaded}

    @staticmethod
    def _collect_dataframe_leaves(value: Any) -> list[pd.DataFrame]:
        '''
        Recursively collect DataFrame leaves from a nested object.
        '''
        if isinstance(value, pd.DataFrame):
            return [value]
        if isinstance(value, dict):
            leaves = []
            for child in value.values():
                leaves.extend(StandaloneExperimentEvents._collect_dataframe_leaves(child))
            return leaves
        return []

    @property
    def df(self) -> pd.DataFrame | dict:
        '''
        Return one event DataFrame, concatenated event DataFrames, or dfs_dict.
        '''
        event_dict = self.dfs_dict.get(self.device_type, self.dfs_dict)
        leaves = self._collect_dataframe_leaves(event_dict)

        if len(leaves) == 1:
            return leaves[0]

        if len(leaves) > 1 and self.concat_files:
            return pd.concat(leaves).sort_index()

        return event_dict


ExperimentEvents = _ExperimentEvents or StandaloneExperimentEvents

__all__ = [
    'ExperimentEvents',
    'StandaloneExperimentEvents',
    '_flatten_nested_dict',
]

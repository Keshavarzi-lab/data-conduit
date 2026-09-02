"""
Readers Module for io subpackage.
--------------------------------

Description:
    Named reader functions for common file formats.
    Readers follow the following: (Path, **kwargs) -> pd.DataFrame.
    More complex readers that return multiple DataFrames (e.g. split_jsonl, split_yaml)
    are provided as standalone functions, but are not registered due to not following the
    standard single-DataFrame output convention required by "collect_dfs".

Contents:
--------------------------------
- add_reader:
    Adds new reader function to registry with a specified name.
    Reader functions must follow the signature:
    (file_path: str | Path, **kwargs) -> pd.DataFrame

- get_reader:
    Retrieves a reader function from the registry by name.
    Raises KeyError if the specified name is not found in the registry.

- read_csv:
    Read a CSV file into a DataFrame.

- read_json:
    Read a JSON/JSONL file into a DataFrame.

- split_jsonl:
    Read a JSONL file and split into metadata + trials DataFrames.

- split_yaml:
    Read a YAML file and split into metadata + trials DataFrames.
"""


################################################################################
# Imports
################################################################################

import json
from pathlib import Path

import pandas as pd
import yaml

################################################################################


################################################################################
# Reader Registry
################################################################################


# ===============================================================================
# 1| Registry
# ===============================================================================

_READERS: dict[str, callable] = {}

# ===============================================================================


# ===============================================================================
# 2| Add Extractor to Registry
# ===============================================================================


def add_reader(
    name: str,
    reader_fn: callable,
) -> None:
    """
    Add a new reader function to the registry with a specified name.

    Parameters:
        name (str):
            The name to register the reader function under (e.g., 'csv', 'json').
            Note: this name is used to reference the reader when calling "collect_dfs".
            Name should be unique and descriptive of the file format or reading method.
        reader_fn (callable):
            The reader function to register. Must follow the signature:
            (file_path: str | Path, **kwargs) -> pd.DataFrame

    Returns:
        None
    """
    if not isinstance(name, str):
        raise TypeError(f"Reader name must be a string, got {type(name)}")
    if not callable(reader_fn):
        raise TypeError(f"Reader function must be callable, got {type(reader_fn)}")
    if name in _READERS:
        raise ValueError(f"A reader with the name '{name}' is already registered.")

    _READERS[name] = reader_fn


# ===============================================================================


# ===============================================================================
# 3| Get Reader from Registry
# ===============================================================================


def get_reader(name: str) -> callable:
    """
    Retrieve a reader function from the registry by name.

    Parameters:
        name (str):
            The name of the reader function to retrieve (e.g., 'csv', 'json').

    Returns:
        callable:
            The reader function registered under the specified name.

    Raises:
        KeyError: If no reader is registered under the specified name.
    """
    if not isinstance(name, str):
        raise TypeError(f"Reader name must be a string, got {type(name)}")
    if name not in _READERS:
        raise KeyError(f"""
                       No reader found with the name '{name}'. 
                       \n
                       Available readers: {list(_READERS.keys())}""")

    return _READERS[name]


# ===============================================================================


################################################################################


################################################################################
# Built-In Readers
################################################################################
"""
Built-in readers for common file formats.
Each reader follows the signature: (file_path: str | Path, **kwargs) -> pd.DataFrame.
These readers are automatically registered when this module is imported.
"""

# ===============================================================================
# 1| CSV Reader
# ===============================================================================


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Uses following as default kwargs if no kwargs are provided:
        - header=0, dtype=str, engine='python'
        - sep splits on commas not inside braces
        - first column as index


    Parameters
        path : str or Path
            Path to the CSV file.
        **kwargs
            Keyword arguments passed directly to pd.read_csv.
            If no kwargs are provided, defaults are used.

    Returns
        pd.DataFrame
            The loaded DataFrame.
    """
    if not kwargs:
        kwargs = {
            "header": 0,
            "dtype": str,
            "engine": "python",
            "sep": r",(?![^{]*})",
            "index_col": 0,
        }
    return pd.read_csv(path, **kwargs)


# ===============================================================================


# ===============================================================================
# 2| JSON / JSONL Reader
# ===============================================================================


def read_json(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Read a JSON or JSONL file into a pandas DataFrame.

    Uses sensible defaults if no kwargs are provided:
        - lines=True (JSONL format)
        - dtype=str, orient='records'

    Parameters
        path : str or Path
            Path to the JSON/JSONL file.
        **kwargs
            Keyword arguments passed directly to pd.read_json.
            If no kwargs are provided, defaults are used.
    Returns
        pd.DataFrame
            The loaded DataFrame.
    """
    if not kwargs:
        kwargs = {
            "lines": True,
            "dtype": str,
            "orient": "records",
            "convert_dates": False,
            "precise_float": True,
        }
    return pd.read_json(path, **kwargs)


# ===============================================================================


# ===============================================================================
# Register Built-In Extractors
# ===============================================================================

add_reader("csv", read_csv)
add_reader("json", read_json)
# ===============================================================================


################################################################################


################################################################################
# Complex Readers (Not Registered)
################################################################################
"""
Complex readers that return multiple DataFrames from a single file.
These do NOT follow the single-DataFrame contract and are therefore 
not registered. They are called directly by consumers (e.g. filetypes).
"""


# ===============================================================================
# 1| Split JSONL to DataFrames
# ===============================================================================


def split_jsonl(
    path: str | Path,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a JSONL file and split into metadata and trials DataFrames.
    Each JSONL record produces one metadata row and multiple trial rows.

    Parameters
    ----------
    path : str or Path
        Path to the JSONL file.
    verbose : bool
        If True, prints detailed output during loading.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (metadata_df, trials_df).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The specified JSONL file does not exist: {path}")

    with open(path) as f:
        records = [json.loads(line) for line in f]

    # Metadata: flatten each record's metadata into one row per record
    meta_df = pd.json_normalize([r["value"]["metadata"] for r in records])

    # Trials: collect all trials across all records, tag each with its timestamp
    all_trials = []
    for r in records:
        for trial in r["value"]["trials"]:
            trial["seconds"] = r.get("seconds")
            all_trials.append(trial)
    trials_df = pd.json_normalize(all_trials)

    if verbose:
        print(f"Loaded {len(records)} records")
        print(f"Metadata: {meta_df.shape}, Trials: {trials_df.shape}")

    return meta_df, trials_df


# ===============================================================================


# ===============================================================================
# 2| Split YAML to DataFrames
# ===============================================================================


def split_yaml(
    path: str | Path,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a YAML file and split into metadata and trials DataFrames.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.
    verbose : bool
        If True, prints detailed output during loading.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (metadata_df, trials_df).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The specified YAML file does not exist: {path}")

    with open(path) as f:
        record = yaml.safe_load(f)

    # Metadata
    meta_df = pd.json_normalize(record["metadata"])

    # Trials
    trials_list = []
    for trial in record["trials"]:
        flat_trial = pd.json_normalize(trial).iloc[0]
        trials_list.append(flat_trial)
    trials_df = pd.DataFrame(trials_list)

    if verbose:
        print(f"Metadata shape: {meta_df.shape}, Trials shape: {trials_df.shape}")

    return meta_df, trials_df


# ===============================================================================


################################################################################

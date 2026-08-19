'''
Core I/O functions used across data-conduit for loading and reading data. 
--------------------------------------------------------------------------

Description: 
    Core functions for file discovery, reading, and collection into nested dictionaries
    of DataFrames. Provides a universal entry point (collect_dfs) for loading data from
    directory trees or pre-built path dictionaries.

    
Contents:
--------------------------------
- collect_folders:      Collect folders from a base path, optionally filtered by prefix.
- collect_file_paths:   Walk a directory tree and build a nested dict of file Paths.
- read_files:           Walk a nested dict of Paths and apply a reader to produce DataFrames.
- collect_dfs:          Master function chaining file discovery, filtering, reading, and output shaping.

'''





################################################################################
# Imports
################################################################################

from pathlib import Path                                # noqa
from typing import Any, Callable                      # noqa

from data_conduit._refactor_template.core.utils import (
    _flatten_nested_dict,
    _parse_selectors,
    _walk_dirtree,
) 

################################################################################






################################################################################
# Collect Folders
################################################################################

def collect_folders(
        base_path: str | Path,
        folder_prefix: str | None = None,
) -> list[Path]:
    """
    Collect folders from a base path that match a specified prefix. 
    Only collects folders one level deep (i.e., does not search subdirectories).
    Only collects directories, not files.

    Parameters
    ----------
    base_path : str | Path
        The base directory to search for folders.
    folder_prefix : str | None
        A prefix to filter folders. Only folders starting with this prefix
        will be collected. If None, all folders in the base path are collected.

    Returns
    -------
    list[Path]
        A list of Path objects representing the collected folders.
    """
    base_path = Path(base_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(
            f"The specified base path does not exist: {base_path}"
        )
    if not base_path.is_dir():
        raise NotADirectoryError(
            f"The specified base path is not a directory: {base_path}"
        )

    pattern = f"{folder_prefix}*" if folder_prefix is not None else "*"
    folders = sorted(
        [f for f in base_path.glob(pattern) if f.is_dir()],
        key=lambda p: p.name,
    )
    if not folders:
        raise FileNotFoundError(
            f"No subfolders found in {base_path}"
            + (f" matching prefix '{folder_prefix}'" if folder_prefix else "")
        )
    return folders

################################################################################






################################################################################
# Collect DataFrames (Master Function)
################################################################################
def collect_dfs(
        base_path: str | Path,
        readers: dict[str, Callable]| None = None,
        reader_kwargs: dict[str, dict] | None = None,
        keep_empty: bool = True,
        flatten: bool = False,
        separator: str = ':',
        verbose: bool = False,
        **kwargs
) -> dict:
    '''
    Collect files from a directory tree into a nested dictionary of DataFrames.

    Walks the entire directory tree from base_path, building a nested dict
    that mirrors the folder structure. Files with extensions matching a
    provided reader are read into DataFrames. Files without a matching
    reader are skipped. If no readers are provided, all files are stored 
    as Paths.

    Level selectors filter at any depth during the walk, supporting exact 
    match lists, single strings, callables, and None (wildcard). Wrappers 
    can map human-readable arguments to level selectors internally.

    Parameters
    ----------
    base_path : str or Path
        Root directory to walk.
    readers : dict[str, Callable] or None
        Mapping of file extensions to reader functions.
        Each reader must follow: (file_path, **kwargs) -> pd.DataFrame.
        e.g. {'.bin': read_harp_bin, '.csv': read_csv}
        If None, all files are stored as Paths.
    reader_kwargs : dict[str, dict] or None
        Mapping of file extensions to kwargs dicts for each reader.
        e.g. {'.bin': {'harp_reader': r, 'addr_to_name': m}}
        Extensions not in this dict receive no extra kwargs.
    keep_empty : bool
        If True, preserve empty subdirectories as empty dicts.
        Default is True.
    flatten : bool
        If True, return a flat dict with keys joined by separator.
    separator : str
        Separator for flattened keys. Default is ':'.
    verbose : bool
        If True, prints warnings during processing.
    **kwargs
        Level selectors for filtering during the walk.
        Must follow naming convention l{n}_selector where n is the 
        depth level (0-indexed from base_path).
        Values can be:
        - None: wildcard, matches everything at this level
        - list: key must be in the list
        - str: exact match on a single key
        - callable: receives key (str), returns True to keep.
          See utils helper callables: starts_with, ends_with, contains.

    Returns
    -------
    dict
        Nested or flat dictionary of DataFrames (or Paths if no readers).
 
    '''
    

    #=== i| Input Validation
    base_path = Path(base_path)
    if not base_path.is_dir():
        raise NotADirectoryError(
            f"The specified base path is not a directory: {base_path}"
        )
    if readers is None:
        readers = {}
    if reader_kwargs is None:
        reader_kwargs = {}
    
    level_selectors = _parse_selectors(kwargs)
    valid_extensions = set(readers.keys()) if readers else None

    #=== ii| Walk Directory Tree and Read Files

    dfs_dict = _walk_dirtree(
        path = base_path,
        depth= 0,
        level_selectors = level_selectors,
        valid_extensions = valid_extensions,
        readers = readers,
        reader_kwargs = reader_kwargs,
        keep_empty = keep_empty,
        verbose = verbose,
    )
    if not dfs_dict and verbose:
            print("Warning: No files found.")

    #=== iii| Flatten if Requested
    if flatten:
        dfs_dict = _flatten_nested_dict(dfs_dict, separator=separator)
        
    return dfs_dict


################################################################################








################################################################################
# Collect File Paths
################################################################################

def collect_file_paths(
        base_path: str | Path,
        file_pattern: str = '*',
        depth: int | None = None,
        _current_depth: int = 0,
) -> dict:
    '''
    Walk a directory tree and build a nested dictionary of file Paths.
    
    Directory names become dictionary keys at each level. Files matching 
    the pattern become leaf values, keyed by their stem (filename without extension).
    
    Parameters
    ----------
    base_path : str or Path
        The root directory to walk.
    file_pattern : str
        Glob pattern for matching files (e.g. '*.csv', '*.bin', '*').
        Only applied to files, not directories.
    depth : int or None
        Maximum number of directory levels to descend into.
        None means unlimited depth.
        0 means only files directly in base_path (no subdirectories).
        1 means files in base_path and one level of subdirectories, etc.
    _current_depth : int
        Internal recursion tracker. Do not set manually.

    Returns
    -------
    dict
        A nested dictionary where:
        - Directory names are keys leading to nested dicts
        - File stems are keys leading to Path values
        
    Examples
    --------
    Given directory structure:
        experiment/
            siteA/
                data.csv
                notes.csv
            siteB/
                session1/
                    recording.csv
    
    >>> collect_file_paths('experiment', '*.csv')
    {
        'siteA': {
            'data': Path('experiment/siteA/data.csv'),
            'notes': Path('experiment/siteA/notes.csv'),
        },
        'siteB': {
            'session1': {
                'recording': Path('experiment/siteB/session1/recording.csv'),
            }
        }
    }

    >>> collect_file_paths('experiment', '*.csv', depth=0)
    {}  # No csv files directly in experiment/
    '''
    base_path = Path(base_path)
    if not base_path.is_dir():
        raise NotADirectoryError(
            f"The specified base path is not a directory: {base_path}"
        )

    result = {}

    #=== Collect matching files at this level
    for item in sorted(base_path.iterdir(), key=lambda p: p.name):
        if item.is_file() and item.match(file_pattern):
            result[item.stem] = item

    #=== Recurse into subdirectories (if depth allows)
    if depth is None or _current_depth < depth:
        for item in sorted(base_path.iterdir(), key=lambda p: p.name):
            if item.is_dir():
                child = collect_file_paths(
                    base_path=item,
                    file_pattern=file_pattern,
                    depth=depth,
                    _current_depth=_current_depth + 1,
                )
                if child:                       # Only include non-empty branches
                    result[item.name] = child

    return result


################################################################################






################################################################################
# Read Files
################################################################################

def read_files(
        paths_dict: dict,
        reader_fn: Callable | None = None,
        reader_kwargs: dict | None = None,
        verbose: bool = False,
) -> dict:
    '''
    Walk a nested dictionary of file Paths and apply a reader function to each,
    producing a nested dictionary of DataFrames with the same structure.

    Parameters
    ----------
    paths_dict : dict
        A nested dictionary where leaf values are Path objects (as produced by collect_file_paths).
    reader_fn : callable or None
        A function with signature (path: Path, **kwargs) -> pd.DataFrame.
        If None, leaves are returned as-is (passthrough mode).
    reader_kwargs : dict or None
        Extra keyword arguments passed to reader_fn for every file.
    verbose : bool
        If True, prints warnings when a reader fails on a file.

    Returns
    -------
    dict
        A nested dictionary with the same key structure as paths_dict, 
        but with DataFrames (or reader output) as leaf values.
    '''
    if reader_kwargs is None:
        reader_kwargs = {}

    result = {}

    for key, value in paths_dict.items():

        #== Nested dict → recurse
        if isinstance(value, dict):
            child = read_files(value, reader_fn, reader_kwargs, verbose)
            if child:
                result[key] = child

        #== Leaf value → apply reader
        else:
            if reader_fn is None:
                result[key] = value
            else:
                try:
                    df = reader_fn(value, **reader_kwargs)
                    if df is not None:
                        result[key] = df
                except Exception as e:
                    if verbose:
                        print(f"Warning: Reader failed for '{key}' ({value}): {e}")

    return result


################################################################################
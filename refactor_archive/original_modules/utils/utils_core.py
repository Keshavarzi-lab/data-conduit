'''
Utils subpackage for data-conduit.

Contents:
--------------------------------

1| Private Helper Functions

    1.1| _flatten_nested_dict: 
            Recursively flattens arbitrarily nested dictionaries of DataFrames or DataArrays.
    
    1.2| _concat_split_dataframes:
            Collapse a (possibly nested) dict of per-file DataFrames into one DataFrame.

2| _get_nested_dict_depth:
        Return the maximum nesting depth of a dictionary (1 = flat dict with no nested dicts).

3| _apply_level_selectors:
        Return a filtered copy of `d`, keeping only keys listed in `selectors[depth]` at each depth.

4| Check Key Matches Selector

    4.1| starts_with(prefix):
            Return true if the key starts with the given prefix.
    
    4.2| ends_with(suffix):
            Return true if the key ends with the given suffix.
    
    4.3| contains(substring):
            Return true if the key contains the given substring.
    
    4.4| exclude(*names):
            Return true if the key is not in the given list of names.
    
    4.5| _matches_selector(key, selector):
            Check if a key matches a selector. Callable is used to allow for flexible matching logic 
            (e.g., regex, custom functions) beyond simple list or string matching. 
            If selector is None, it matches everything.

5| _parse_selectors:
        Validate and parse level selector kwargs into a dictionary mapping depth levels to their corresponding selector values.

6| _is_readable:
        Check if file has matching reader. If no readers defined, read everything.

7| _passes_selector:
        Check if name passes selector at given level. If no selector for that level, pass everything.

8| _attempt_read:
        Attempt to read file with matching reader. Returns DataFrame, Path, or None if no reader matches or read fails.

9| _walk_dirtree:
        Walk Directory Tree and Collect DataFrames/DataArrays. 

        


'''





################################################################################
# Imports
################################################################################

import warnings
from collections.abc import Callable, Collection
from typing import Any

import pandas as pd
import xarray as xr

################################################################################






################################################################################
# Private Helper Functions
################################################################################





#===============================================================================
# 1| Flatten Nested Dictionary of DataFrames/DataArrays
#===============================================================================
def _flatten_nested_dict(
                        dfs_dict: dict,
                        parent_key: str = '',
                        separator: str = ':'
                        ) -> dict[str, pd.DataFrame | xr.DataArray]:
    '''
    Recursively flattens arbitrarily nested dictionaries of DataFrames or DataArrays.
    
    Keys are concatenated using the specified separator to create unique identifiers 
    for each DataFrame/DataArray in the flattened structure.
    
    ----------
    Parameters:
        dfs_dict (dict): 
            The nested dictionary to flatten.
        parent_key (str): 
            The base key to use for the current level of recursion.
        separator (str): 
            The string used to separate keys in the flattened dictionary.
    Returns:
        dict: 
            A flattened dictionary where keys are the concatenated path of the original 
            nested keys, and values are the DataFrames or DataArrays.
    '''
    
    items = {}
    for key, value in dfs_dict.items():
        new_key = f'{parent_key}{separator}{key}' if parent_key else str(key)
        if isinstance(value, dict):
            items.update(_flatten_nested_dict(value, new_key, separator))
        else:
            items[new_key] = value
    return items

#===============================================================================


#===============================================================================
# 1b| Concatenate a Split (Multi-File) Stream Into One DataFrame, Filename-First
#===============================================================================
def _concat_split_dataframes(
        df_or_dict,
        *,
        on_rollback: str = 'warn',
):
    '''
    Collapse a (possibly nested) dict of per-file DataFrames into one DataFrame.

    A data-conduit FileType reader returns a plain DataFrame for a single-file
    stream, but a (possibly nested) dict of DataFrames when a stream was split
    across several files (for example an ExperimentEvents CSV that rolled over
    mid-session, or several VideoData segments). Several call sites need exactly
    one DataFrame per stream, so this collapses any such dict into one. It is the
    single home of the filename-first policy that used to be copied per call site.

    Ordering policy (deliberate):
        The pieces are joined by FILENAME FIRST, then by the Time index WITHIN
        each file. The dict is walked in sorted key order (the keys are file
        stems whose zero-padded ISO timestamps make lexical order chronological),
        each file's own rows are sorted by Time, and the files are concatenated
        in that filename order. The combined frame is NOT sorted globally by Time.
        That matters because a stream's clock can reset toward 0 between files in
        edge cases: a global time sort would hoist the later file's small
        timestamps in front of the earlier file and silently MASK the rollback,
        fabricating a monotonic-but-wrong index. Keeping filename order leaves a
        reset visible as a non-monotonic Time index, surfaced per ``on_rollback``.

    ----------
    Parameters:
        df_or_dict (pandas.DataFrame | dict | object):
            Either a DataFrame (its rows are sorted by Time and returned), a
            (possibly nested) dict of DataFrames (ordered by filename, each
            sorted by Time within itself, then concatenated), or any other
            object (returned unchanged so non-DataFrame readers are unaffected).
        on_rollback (str):
            What to do when the concatenated Time index is not monotonic
            increasing (the clock reset between files): ``'warn'`` (default) to
            warn, ``'error'`` to raise, or ``'ignore'`` to do neither (used when
            a later step performs its own rollback check).
    Returns:
        pandas.DataFrame | object:
            One DataFrame for the DataFrame/dict cases, otherwise the input as-is.
    '''

    # A single file: order its own rows by the Time index so the time logs WITHIN
    # this file are ordered. A within-file sort is safe (one continuous write); it
    # is only the CROSS-file global sort that we avoid (see the dict branch).
    if isinstance(df_or_dict, pd.DataFrame):
        return df_or_dict.sort_index()

    # A multi-file stream arrives as a dict keyed by file stem. Walk it in sorted
    # KEY (filename) order so the files join chronologically, flattening each value
    # first (the recursion also handles any nested dict), then concatenate in that
    # order with NO global time sort.
    if isinstance(df_or_dict, dict):
        frames = [
            _concat_split_dataframes(value, on_rollback=on_rollback)
            for _, value in sorted(df_or_dict.items(), key=lambda kv: kv[0])
        ]
        combined = pd.concat(frames)
        # After ordering by filename, the Time index can only DECREASE if the clock
        # actually reset between files, so a non-monotonic result is the rollback
        # showing through. Surface it per the caller's policy.
        if on_rollback != 'ignore' and not combined.index.is_monotonic_increasing:
            message = (
                'multi-file stream has a non-monotonic Time index after ordering by '
                'filename; the recording clock appears to have reset between files. '
                'Rows are kept in filename order (not globally time-sorted) so the '
                'reset stays visible; downstream steps assuming monotonic time may '
                'need attention.'
            )
            if on_rollback == 'error':
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
        return combined

    # Anything else (should not happen for the FileType readers) is passed through.
    return df_or_dict

#===============================================================================



#===============================================================================
# 2| Get Maximum Depth of Nested Dictionary
#===============================================================================
def _get_nested_dict_depth(d: dict) -> int:
    """Return the maximum nesting depth of a dictionary (1 = flat dict with no nested dicts)."""
    if not isinstance(d, dict) or not d:
        return 0
    return 1 + max(_get_nested_dict_depth(v) for v in d.values())

#===============================================================================


#===============================================================================
# 3| Apply Level-Based Selectors to Nested Dictionary
#===============================================================================
def _apply_level_selectors(
        d: dict,
        selectors: dict[int, list],
        current_depth: int = 0,
) -> dict:
    """Return a filtered copy of `d`, keeping only keys listed in `selectors[depth]` at each depth."""
    result = {}
    for key, value in d.items():
        if current_depth in selectors and not _matches_selector(key, selectors[current_depth]):
            continue
        if isinstance(value, dict):
            filtered = _apply_level_selectors(value, selectors, current_depth + 1)
            if filtered:
                result[key] = filtered
        else:
            result[key] = value
    return result

#===============================================================================


#===============================================================================
# 4| Check Key Matches Selector
#===============================================================================

def starts_with(prefix: str) -> Callable[[str], bool]:
    """Return a function that checks if a string starts with the given prefix."""
    return lambda key: key.startswith(prefix)

def ends_with(suffix: str) -> Callable[[str], bool]:
    """Return a function that checks if a string ends with the given suffix."""
    return lambda key: key.endswith(suffix)

def contains(substring: str) -> Callable[[str], bool]:
    """Return a function that checks if a string contains the given substring."""
    return lambda key: substring in key

def exclude(*names: str) -> Callable[[str], bool]:
    """Return a selector that keeps every key EXCEPT the named ones.

    This is the complement of passing a list (which *includes* only the named
    keys). It is the natural way to express "all subfolders except these":
    ``l0_selector=exclude('skip_me', 'and_me')`` keeps everything else. The
    returned function is used exactly like ``starts_with``/``contains`` — it is
    handed to a selector and called per key by ``_matches_selector``.
    """
    # Build a set once for fast membership tests, then keep keys not in it.
    blocked = set(names)
    return lambda key: key not in blocked


def _matches_selector(
        key: str,
    selector: list | tuple | set | frozenset | Callable | str | None
)-> bool:
    '''
    Check if a key matches a selector. 
    Callable is used to allow for flexible matching logic (e.g., regex, custom functions) beyond simple list or string matching.
    If selector is None, it matches everything.

    Base helper callables:
        - starts_with(prefix):
            Return true if the key starts with the given prefix.
        - ends_with(suffix):
            Return true if the key ends with the given suffix.
        - contains(substring):
            Return true if the key contains the given substring.

    Parameters
        key : str
            The key to check (folder name or file stem).
        selector : list, tuple, set, frozenset, callable, str, or None
            - None: matches everything (wildcard)
            - list, tuple, set, frozenset: key must be in the collection
            - callable: must return True for the key
            - str: exact match

    Returns
        bool:
            True if the key matches the selector, False otherwise.
    '''
    if selector is None:
        return True
    
    if callable(selector):         
        return selector(key)
    if isinstance(selector, (list, tuple, set, frozenset)):
        return key in selector
    if isinstance(selector, str):
        return key == selector

    else:
        raise ValueError(f"Invalid selector type: {type(selector)}. Must be list, tuple, set, frozenset, callable, str, or None.")

#=============================================================================== 


#===============================================================================
# 5| Parse Selectors
#===============================================================================
def _parse_selectors(
        kwargs: dict
) -> dict[int, any]:
    '''
    Validate and parse level selector kwargs into a dictionary mapping depth levels to their corresponding selector values.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments from collect_dfs. Must follow the pattern 'level_{n}_selector' where n is a non-negative integer.
    Returns
    -------
    dict[int, any]
        Mapping of {depth_level: selector_value} for each provided level selector.
        E.g. {'l0_selector': ['folder1', 'folder2'], 'l1_selector': lambda x: x.startswith('data_')} -> {0: ['folder1', 'folder2'], 1: lambda x: x.startswith('data_')}
   
    '''

    #=== 0| Check if level_selectors is empty or None
    if not kwargs:
        return {}
    
    #=== 1| Validate Selector Naming
    for key in kwargs:
        if not (key.startswith('l') and key.endswith('_selector')):
            raise ValueError(f'''
                             Invalid keyword argument: '{key}'.
                             Selectors must follow 'l{{n}}_selector' pattern, where n is a non-negative integer 
                             (e.g., 'l0_selector', 'l1_selector').
                '''
            )
        try:
            level = int(key[1:-9]) # Extract n from 'l{n}_selector' and check if it's an integer
        except ValueError as e:
            raise ValueError(f'''
                             Invalid selector key: '{key}'. 
                             The part between 'l' and '_selector' must be a non-negative integer (e.g., 'l0_selector', 'l1_selector').
                '''
            ) from e

        if level < 0:
            raise ValueError(f'''
                             Invalid selector key: '{key}'. 
                             The part between 'l' and '_selector' must be a non-negative integer (e.g., 'l0_selector', 'l1_selector').
                '''
            )
        
    return {int(key[1:-9]): value for key, value in kwargs.items()}

#=============================================================================== 


#===============================================================================
# 6| Check that file has matching reader
#===============================================================================
def _is_readable(item,
                 valid_extensions
                ):
    '''Check if file has matching reader. If no readers defined, read everything.'''
    
    if not valid_extensions:
        return True
    return item.suffix in valid_extensions


#=============================================================================== 


#===============================================================================
# 7| Check if name passes selector at given level
#===============================================================================
def _passes_selector(name,
                     depth,
                     level_selectors
                     ):
    '''Check if name passes selector at given level. If no selector for that level, pass everything.'''
    if depth not in level_selectors:
        return True
    return _matches_selector(name, level_selectors[depth])


#=============================================================================== 


#===============================================================================
# 8| Attempt to read file
#===============================================================================
def _attempt_read(item,
                  readers,
                  reader_kwargs,
                  verbose
                  ):
    '''Attempt to read file with matching reader. Returns DataFrame, Path, or None if no reader matches or read fails.'''
    if item.suffix not in readers:
        return item
    
    reader_func = readers[item.suffix]
    extra = reader_kwargs.get(item.suffix, {})
    try:
        return reader_func(item, **extra)
    except Exception as e:
        if verbose:
            print(f"Failed to read {item} with reader for {item.suffix}: {e}")
        return None


#=============================================================================== 


#===============================================================================
# 9| Walk Directory Tree and Collect DataFrames/DataArrays
#===============================================================================
def _walk_dirtree(
        path,
        depth,
        level_selectors,
        valid_extensions,
        readers,
        reader_kwargs,
        keep_empty,
        verbose
):
    
    result = {}

    for item in sorted(path.iterdir(), key = lambda p: p.name):            # Sort items to enforce consistent order

        if item.is_dir():
            if depth in level_selectors and not _matches_selector(key = item.name, selector = level_selectors[depth]):
                    continue
            
            sub_result = _walk_dirtree(
                path = item,
                depth = depth + 1,
                level_selectors = level_selectors,
                valid_extensions = valid_extensions,
                readers = readers,
                reader_kwargs = reader_kwargs,
                keep_empty = keep_empty,
                verbose = verbose
                )
            
            if sub_result or keep_empty is True:                                   # Only include non-empty folders unless keep_empty is True
                result[item.name] = sub_result
        
        elif item.is_file():
           
            if not _is_readable(item, valid_extensions):                   # Skip files that don't have a matching reader (if valid_extensions is defined)
                continue
            if not _passes_selector(item.stem, depth, level_selectors):    # Skip files that don't pass the selector for their depth level (if defined)
                continue

            
            df = _attempt_read(
                item = item,
                readers = readers,
                reader_kwargs = reader_kwargs,
                verbose = verbose
            )
            if df is not None:
                result[item.stem] = df
        
    return result




#=============================================================================== 


    

#===============================================================================
# 10| 
#===============================================================================



#=============================================================================== 

################################################################################
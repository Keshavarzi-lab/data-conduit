"""Shared nested-data, selector, and directory-walking utilities."""

import warnings
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

Selector = list[str] | tuple[str, ...] | set[str] | frozenset[str] | Callable[[str], bool] | str | None


def _flatten_nested_dict(
    dfs_dict: dict,
    parent_key: str = "",
    separator: str = ":",
) -> dict[str, pd.DataFrame | xr.DataArray]:
    """Flatten a nested dictionary, joining its key path with ``separator``."""
    items: dict[str, pd.DataFrame | xr.DataArray] = {}
    for key, value in dfs_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(_flatten_nested_dict(value, new_key, separator))
        else:
            items[new_key] = value
    return items


def _concat_split_dataframes(
    df_or_dict: Any,
    *,
    on_rollback: str = "warn",
) -> Any:
    """Collapse a nested dictionary of per-file frames into one DataFrame.

    Pieces are ordered by filename/key first and by their time index only within
    each file. The combined frame is deliberately not globally time-sorted: a
    clock reset between files must remain visible as a non-monotonic index.

    ``on_rollback`` may be ``"warn"`` (the default), ``"error"``, or
    ``"ignore"``. Non-DataFrame, non-dictionary values pass through unchanged.
    """
    if isinstance(df_or_dict, pd.DataFrame):
        return df_or_dict.sort_index()

    if isinstance(df_or_dict, dict):
        frames = [_concat_split_dataframes(value, on_rollback=on_rollback) for _, value in sorted(df_or_dict.items(), key=lambda item: item[0])]
        combined = pd.concat(frames)
        if on_rollback != "ignore" and not combined.index.is_monotonic_increasing:
            message = (
                "multi-file stream has a non-monotonic Time index after ordering by "
                "filename; the recording clock appears to have reset between files. "
                "Rows are kept in filename order (not globally time-sorted) so the "
                "reset stays visible; downstream steps assuming monotonic time may "
                "need attention."
            )
            if on_rollback == "error":
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
        return combined

    return df_or_dict


def _get_nested_dict_depth(d: dict) -> int:
    """Return the maximum dictionary nesting depth, with an empty dict at zero."""
    if not isinstance(d, dict) or not d:
        return 0
    return 1 + max(_get_nested_dict_depth(value) for value in d.values())


def _apply_level_selectors(
    d: dict,
    selectors: Mapping[int, Selector],
    current_depth: int = 0,
) -> dict:
    """Return a copy filtered by the selector configured at each depth."""
    result = {}
    for key, value in d.items():
        if current_depth in selectors and not _matches_selector(
            key,
            selectors[current_depth],
        ):
            continue
        if isinstance(value, dict):
            filtered = _apply_level_selectors(value, selectors, current_depth + 1)
            if filtered:
                result[key] = filtered
        else:
            result[key] = value
    return result


def starts_with(prefix: str) -> Callable[[str], bool]:
    """Return a selector that matches keys starting with ``prefix``."""
    return lambda key: key.startswith(prefix)


def ends_with(suffix: str) -> Callable[[str], bool]:
    """Return a selector that matches keys ending with ``suffix``."""
    return lambda key: key.endswith(suffix)


def contains(substring: str) -> Callable[[str], bool]:
    """Return a selector that matches keys containing ``substring``."""
    return lambda key: substring in key


def exclude(*names: str) -> Callable[[str], bool]:
    """Return a selector that matches every key except the named keys."""
    blocked = set(names)
    return lambda key: key not in blocked


def _matches_selector(key: str, selector: Selector) -> bool:
    """Return whether ``key`` matches a supported selector."""
    if selector is None:
        return True
    if callable(selector):
        return selector(key)
    if isinstance(selector, str):
        return key == selector
    if isinstance(selector, (list, tuple, set, frozenset)):
        return key in selector
    raise ValueError(f"Invalid selector type: {type(selector)}. Must be list, tuple, set, frozenset, callable, str, or None.")


def _parse_selectors(kwargs: Mapping[str, Any]) -> dict[int, Any]:
    """Validate ``l{n}_selector`` keys and return selectors by depth level."""
    if not kwargs:
        return {}

    parsed: dict[int, Any] = {}
    for key, value in kwargs.items():
        if not (key.startswith("l") and key.endswith("_selector")):
            raise ValueError(
                f"Invalid keyword argument: {key!r}. Selectors must follow the 'l{{n}}_selector' pattern, where n is a non-negative integer (e.g. 'l0_selector', 'l1_selector')."
            )
        try:
            level = int(key[1:-9])
        except ValueError as exc:
            raise ValueError(f"Invalid selector key: {key!r}. The part between 'l' and '_selector' must be a non-negative integer.") from exc
        if level < 0:
            raise ValueError(f"Invalid selector key: {key!r}. The part between 'l' and '_selector' must be a non-negative integer.")
        parsed[level] = value
    return parsed


def _is_readable(item: Path, valid_extensions: Collection[str] | None) -> bool:
    """Return whether a path has a supported extension."""
    if not valid_extensions:
        return True
    return item.suffix in valid_extensions


def _passes_selector(
    name: str,
    depth: int,
    level_selectors: Mapping[int, Selector],
) -> bool:
    """Return whether a name passes the selector at ``depth``."""
    if depth not in level_selectors:
        return True
    return _matches_selector(name, level_selectors[depth])


def _attempt_read(
    item: Path,
    readers: Mapping[str, Callable[..., Any]],
    reader_kwargs: Mapping[str, dict[str, Any]],
    verbose: bool,
) -> Any:
    """Read a path with its extension's reader, or return the path unchanged."""
    if item.suffix not in readers:
        return item

    reader_func = readers[item.suffix]
    extra = reader_kwargs.get(item.suffix, {})
    try:
        return reader_func(item, **extra)
    except Exception as exc:  # noqa: BLE001 - legacy best-effort traversal contract.
        if verbose:
            print(f"Failed to read {item} with reader for {item.suffix}: {exc}")
        return None


def _walk_dirtree(
    path: Path,
    depth: int,
    level_selectors: Mapping[int, Selector],
    valid_extensions: Collection[str] | None,
    readers: Mapping[str, Callable[..., Any]],
    reader_kwargs: Mapping[str, dict[str, Any]],
    keep_empty: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Walk a directory tree and collect selected, readable file values."""
    result: dict[str, Any] = {}

    for item in sorted(path.iterdir(), key=lambda candidate: candidate.name):
        if item.is_dir():
            if depth in level_selectors and not _matches_selector(
                item.name,
                level_selectors[depth],
            ):
                continue

            sub_result = _walk_dirtree(
                path=item,
                depth=depth + 1,
                level_selectors=level_selectors,
                valid_extensions=valid_extensions,
                readers=readers,
                reader_kwargs=reader_kwargs,
                keep_empty=keep_empty,
                verbose=verbose,
            )
            if sub_result or keep_empty:
                if item.name in result:
                    raise ValueError(f"Directory entries in {path} map to duplicate output key {item.name!r}; file stems and directory names must be unique.")
                result[item.name] = sub_result
            continue

        if not item.is_file() or not _is_readable(item, valid_extensions):
            continue
        if not _passes_selector(item.stem, depth, level_selectors):
            continue

        value = _attempt_read(
            item=item,
            readers=readers,
            reader_kwargs=reader_kwargs,
            verbose=verbose,
        )
        if value is not None:
            if item.stem in result:
                raise ValueError(f"Directory entries in {path} map to duplicate output key {item.stem!r}; file stems and directory names must be unique.")
            result[item.stem] = value

    return result

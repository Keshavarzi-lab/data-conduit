"""Shared selector, nested-data, and directory traversal helpers."""

from .utils_core import (
    _apply_level_selectors,
    _attempt_read,
    _concat_split_dataframes,
    _flatten_nested_dict,
    _get_nested_dict_depth,
    _is_readable,
    _matches_selector,
    _parse_selectors,
    _passes_selector,
    _walk_dirtree,
    contains,
    ends_with,
    exclude,
    starts_with,
)

__all__ = [
    "_apply_level_selectors",
    "_attempt_read",
    "_concat_split_dataframes",
    "_flatten_nested_dict",
    "_get_nested_dict_depth",
    "_is_readable",
    "_matches_selector",
    "_parse_selectors",
    "_passes_selector",
    "_walk_dirtree",
    "contains",
    "ends_with",
    "exclude",
    "starts_with",
]

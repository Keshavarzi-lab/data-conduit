"""I/O helpers for discovering and reading data files."""

from .io_core import collect_dfs, collect_file_paths, collect_folders, read_files
from .readers import (
    add_reader,
    get_reader,
    read_csv,
    read_json,
    split_jsonl,
    split_yaml,
)

__all__ = [
    "add_reader",
    "collect_dfs",
    "collect_file_paths",
    "collect_folders",
    "get_reader",
    "read_csv",
    "read_files",
    "read_json",
    "split_jsonl",
    "split_yaml",
]

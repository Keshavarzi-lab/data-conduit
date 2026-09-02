"""IO subpackage for data-conduit."""

from data_conduit.io.io_core import (
    collect_folders,
    collect_file_paths,
    collect_dfs,
    read_files,
)



from data_conduit.io.readers import (
    add_reader,
    get_reader,
    read_csv,
    read_json,
    split_jsonl,
    split_yaml,
)

__all__ = [
    "add_reader",
    "collect_folders",
    "get_reader",
    "read_csv",
    "read_json",
    "split_jsonl",
    "split_yaml",
    "collect_file_paths",
    "collect_dfs",
    "read_files",
] 
# Create and register a file reader

The low-level reader registry gives a stable name to a callable that converts
one file into a pandas object. A reader must accept a file path, may accept
keyword arguments, and should return a `DataFrame`.

This example adds support for tab-separated files and then passes the registered
callable to `collect_dfs`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data_conduit.core.io import add_reader, collect_dfs, get_reader


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


# Registration is process-wide, so register a given name only once.
try:
    tsv_reader = get_reader("tabular_tsv")
except KeyError:
    add_reader("tabular_tsv", read_tsv)
    tsv_reader = get_reader("tabular_tsv")


with TemporaryDirectory() as directory:
    root = Path(directory)
    pd.DataFrame(
        {"Time": [0.0, 0.5, 1.0], "Value": [2.0, 3.5, 4.0]}
    ).to_csv(root / "measurements.tsv", sep="\t", index=False)

    tables = collect_dfs(
        root,
        readers={".tsv": tsv_reader},
        reader_kwargs={".tsv": {"index_col": "Time"}},
        keep_empty=False,
    )

    print(tables["measurements"])
```

The key in `readers` is the file extension, including its leading dot. The key
in `reader_kwargs` must match it. Files with other extensions are skipped when a
reader mapping is supplied. Leaf keys use the file stem, so
`measurements.tsv` is available as `tables["measurements"]`.

## Filter the directory walk

Level selectors apply to names at a particular depth. Strings perform exact
matching; lists accept any listed name; callables implement custom rules.
Replace the path in this adaptation snippet with one of your session folders.

```python
from data_conduit.core.utils import starts_with

tables = collect_dfs(
    "path/to/session",
    readers={".tsv": tsv_reader},
    l0_selector=starts_with("device-"),
)
```

## File readers and catalog readers are different

A registered file reader handles one file. A `StreamCatalog` reader handles one
session directory and may use several file readers internally:

```python
from data_conduit.datastructures import StreamCatalog


def read_measurements(session_path: Path):
    return collect_dfs(
        session_path,
        readers={".tsv": tsv_reader},
        keep_empty=False,
    )


catalog = StreamCatalog()
catalog.add_reader("measurements", read_measurements)
```

Keep those responsibilities separate: file readers decode a format, while
catalog readers define what one workflow loads from each session.

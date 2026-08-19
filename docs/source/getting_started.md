# Getting started

## Installation

From a source checkout, install the core package for CSV, JSON, YAML, pandas,
and xarray workflows:

```bash
git clone https://github.com/Keshavarzi-lab/data-conduit.git
cd data-conduit
python -m pip install .
```

Optional integrations are installed explicitly. For example:

```bash
python -m pip install ".[harp,pose]"
```

## Load a directory of CSV files

`collect_dfs` walks a directory tree and returns a nested dictionary matching
that tree. Reader functions are selected by file extension.

```python
from pathlib import Path

from data_conduit.core.io import collect_dfs, read_csv

tables = collect_dfs(
    Path("/path/to/experiment"),
    readers={".csv": read_csv},
)
```

For multi-session workflows, start with the {doc}`workflow` guide. It explains
how session selection, a reusable stream catalog, and `DataStructure` fit
together. The {doc}`examples` pages then build each part independently.

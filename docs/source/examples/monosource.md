# Build a `MonoSource`

`MonoSource` keeps the nested dictionary produced by the I/O layer and can expose
selected leaves under stable, human-readable names. It is useful when each
output comes from one location in the source tree.

The example below supplies an in-memory dictionary, so no files are required.

```python
import pandas as pd

from data_conduit.datasources.monosource import MonoSource

events = pd.DataFrame(
    {"Event": ["session started", "cue", "trial ended"]},
    index=pd.Index([0.0, 1.2, 2.4], name="Time"),
)

temperature = pd.DataFrame(
    {"Value": [20.1, 20.2, 20.2]},
    index=pd.Index([0.0, 1.0, 2.0], name="Time"),
)

source = MonoSource(
    dfs_dict={
        "logs": {"events.csv": events},
        "sensors": {"temperature.csv": temperature},
    },
    monosource_data_arrays={
        "events": {
            "l0_selector": "logs",
            "l1_selector": "events.csv",
        },
        "temperature": {
            "l0_selector": "sensors",
            "l1_selector": "temperature.csv",
        },
    },
    verbose=True,
)

print(source.dfs_dict.keys())
print(source.data_arrays["temperature"])
```

Each `monosource_data_arrays` entry uses the same `l{n}_selector` convention as
`collect_dfs`. A single matched DataFrame is converted with
`DataFrame.to_xarray()`. An existing `DataArray` is retained as-is.

## Load from a directory instead

To make the source perform its own file discovery, provide a directory and a
mapping from extensions to reader callables. Replace the path in this adaptation
snippet with one of your session folders:

```python
from data_conduit.core.io import read_csv

source = MonoSource(
    experiment_directory_path="path/to/session",
    readers={".csv": read_csv},
    reader_kwargs={".csv": {"index_col": "Time"}},
    monosource_data_arrays={
        "events": {
            "l0_selector": "logs",
            "l1_selector": "events",
        }
    },
)
```

Passing `dfs_dict` and `experiment_directory_path` are alternative modes. Use a
pre-built dictionary when file discovery has already happened, or a path when
the source should own that step.

## Match more than one leaf

If a friendly-name selector matches several leaves, each output is retained and
its flattened path is appended to the name. This avoids silently discarding an
ambiguous match:

```python
all_logs = MonoSource(
    dfs_dict={
        "logs": {
            "events-a.csv": events,
            "events-b.csv": events.copy(),
        }
    },
    monosource_data_arrays={
        "events": {"l0_selector": "logs"},
    },
    verbose=True,
)

print(list(all_logs.data_arrays))
# ['events:logs:events-a.csv', 'events:logs:events-b.csv']
```

Presets such as `ExperimentEvents` and `VideoData` build on this pattern for
common Bonsai outputs. HARP-specific presets live under
`data_conduit.integrations.harp.datasource_presets`.

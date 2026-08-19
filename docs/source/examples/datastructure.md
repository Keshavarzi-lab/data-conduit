# Build a multi-session `DataStructure`

`StreamCatalog` defines the reusable recipe for one session. `DataStructure`
selects sessions, applies that recipe independently to each one, and combines
like-named results.

The complete example below creates two temporary session folders, so it can run
without external data.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data_conduit.datastructures import DataStructure, StreamCatalog


def read_measurements(session_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        session_path / "measurements.csv",
        index_col="Time",
    )


def add_speed(objects: dict[str, object]) -> dict[str, object]:
    configured = dict(objects)
    measurements = configured["measurements"].copy()
    measurements["speed"] = measurements["position"].diff()
    configured["measurements"] = measurements
    return configured


catalog = StreamCatalog()
catalog.add_reader("measurements", read_measurements)
catalog.add_configurator("add_speed", add_speed)

with TemporaryDirectory() as directory:
    root = Path(directory)

    for session_name, offset in [("session-a", 0.0), ("session-b", 10.0)]:
        session_path = root / session_name
        session_path.mkdir()
        pd.DataFrame(
            {
                "Time": [0.0, 1.0, 2.0],
                "position": [offset, offset + 1.0, offset + 3.0],
            }
        ).to_csv(session_path / "measurements.csv", index=False)

    pipeline = DataStructure(root, catalog, depth=0)

    # Selection performs no reader I/O.
    print(list(pipeline.select()))

    streams = pipeline.load()
    measurements = streams["measurements"]
    print(measurements)
```

The combined DataFrame retains its `Time` index and gains a `session` column.
The configurator ran once per session before the tables were concatenated, so
the first difference in each session remains missing rather than being computed
across a session boundary.

## A catalog is a declaration

Reader and configurator names are available for inspection:

```python
print(catalog.reader_names)
print(catalog.configurator_names)
print(catalog)
```

Readers execute first, in registration order. Configurators then execute in
their own registration order, each receiving the complete mapping returned by
the previous stage.

## Optional sources

Mark a reader optional only when some valid sessions may not contain that
source:

```python
from pathlib import Path

import pandas as pd

from data_conduit.datastructures import StreamCatalog


def read_pose(session_path: Path) -> pd.DataFrame:
    pose_path = session_path / "pose.csv"
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    return pd.read_csv(pose_path, index_col="Time")


optional_catalog = StreamCatalog()
optional_catalog.add_reader("pose", read_pose, optional=True)
```

An optional reader skips only `FileNotFoundError`. Parsing failures, malformed
data, and other exceptions still stop the load. If `pose` exists in only a
subset of sessions, its eventual `StreamContainer` combines that subset without
forcing placeholder pose data into the other sessions.

## Inspect the result and intermediate state

After a successful load, a `DataStructure` behaves like a mapping of stream
names:

```python
print(list(pipeline.keys()))
print(pipeline["measurements"])
print(len(pipeline))
```

It also retains the most recent selection and regrouping state:

```python
print(pipeline.sessions)
print(pipeline.containers)
```

This makes it possible to answer three separate questions: which sessions were
selected, which sessions contributed each stream, and what final combined
objects were produced.

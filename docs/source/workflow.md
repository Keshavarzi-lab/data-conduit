# Building a workflow

The two master components are `StreamCatalog` and `DataStructure`.
`StreamCatalog` describes how to turn one session directory into named streams;
`DataStructure` applies that recipe to a selected group of sessions and combines
the results.

![Data flow through the data-conduit orchestration layer](_static/images/data-flow.svg)

## 1. Write session readers

A reader has one required input: the path to a single selected session. It may
return a `DataFrame`, `DataArray`, `Dataset`, a mapping of those objects, or a
wrapper exposing `.df` or `.data_arrays`.

```python
from pathlib import Path

import pandas as pd


def read_events(session_path: Path) -> pd.DataFrame:
    return pd.read_csv(session_path / "events.csv", index_col="Time")


def read_signal(session_path: Path) -> pd.DataFrame:
    return pd.read_csv(session_path / "signal.csv", index_col="Time")


def read_pose(session_path: Path) -> pd.DataFrame:
    return pd.read_csv(session_path / "pose.csv", index_col="Time")
```

Readers should concern themselves with one source. Cross-source operations,
such as putting pose on video time or deriving trials from an event log, belong
in configurators.

## 2. Declare the per-session catalog

Reader names become stream names for single pandas/xarray objects. A source with
multiple `.data_arrays` produces names such as `pose:position` and
`pose:confidence`.

```python
from data_conduit.datastructures import StreamCatalog

catalog = StreamCatalog()
catalog.add_reader("events", read_events)
catalog.add_reader("signal", read_signal)
catalog.add_reader("pose", read_pose, optional=True)
```

Registration order is retained. Required readers propagate a missing-file
error. An optional reader may skip `FileNotFoundError`, allowing a stream to be
present for only the sessions that contain it.

## 3. Add ordered configurators

A configurator sees every object loaded for one session. Its contract is:

```text
dict[str, object] -> dict[str, object]
```

This is the place to express dependencies between sources. The example below
turns the raw event log into a trial table and removes the intermediate log from
the eventual output.

```python
from data_conduit.datastructures import TrialSpec, parse_trials

trial_spec = TrialSpec(
    closes_trial="Trial ended",
    session_start="Session started",
)


def build_trials(objects: dict[str, object]) -> dict[str, object]:
    configured = dict(objects)
    events = configured.pop("events")
    configured["trials"] = parse_trials(events, trial_spec)
    return configured


catalog.add_configurator("build_trials", build_trials)
```

Configurator order is meaningful: each stage receives the mapping returned by
the previous one. Name each stage after the transformation it performs so the
catalog representation remains useful during inspection.

## 4. Configure session selection

`depth` counts intermediate directory levels between the root and the session
folders. Their names can be retained as provenance with `level_names`.

For this layout, `depth=2`:

```text
experiment/
├── mouse-a/
│   └── day-01/
│       ├── session-001/
│       └── session-002/
└── mouse-b/
    └── day-01/
        └── session-001/
```

```python
from data_conduit.datastructures import DataStructure

pipeline = DataStructure(
    "path/to/experiment",
    catalog,
    depth=2,
    level_names=("mouse", "day"),
    l0_selector=["mouse-a", "mouse-b"],
)
```

The constructor performs no experimental-data I/O. Use `select()` to inspect
the exact `SessionRef` objects before loading:

```python
for session_id, session in pipeline.select().items():
    print(session_id, session.path, session.levels)
```

## 5. Load and combine

`load()` repeats selection, applies the catalog independently to each session,
then regroups and combines like-named streams.

```python
streams = pipeline.load()

trials = streams["trials"]
signal = streams["signal"]
```

DataFrames are concatenated by row. Xarray objects are concatenated along
`Time` by default. Both receive source-session identity, and explicitly named
directory levels are broadcast onto the combined result.

A source missing from one session is not filled into an unrelated stream. Its
container simply consists of the sessions that produced it.

## 6. Slice downstream data by trials

Trial tables and continuous streams carry the same session identity. The slicing
helpers therefore constrain both session and time, preventing equal timestamps
from another session from leaking into a result.

```python
from data_conduit.datastructures import slice_stream_for_trial

one_trial = trials.iloc[0]
trial_signal = slice_stream_for_trial(
    signal,
    one_trial,
    segment="trial",
)
```

For experiment-specific sub-intervals, declare named segments in `TrialSpec` and
request them by name. Scientific filtering—outcome, condition, animal, day—stays
outside the generic slicing machinery: filter the trial table first, then slice
the streams for the retained rows.

## Where the supporting modules fit

- `data_conduit.core.io` and the datasource/integration packages help implement
  readers.
- `data_conduit.core.timestamps`, `data_conduit.core.sync`, and
  `data_conduit.core.globaltimes` help implement alignment configurators.
- `data_conduit.core.virtualarrays` represents and queries values spread across
  physical devices or channels.
- `data_conduit.datastructures.TrialSpec` and the slicing helpers support the
  analysis stage after streams have been combined.

The {doc}`examples <examples>` develop each of these pieces independently
before putting them together in a complete synthetic workflow.

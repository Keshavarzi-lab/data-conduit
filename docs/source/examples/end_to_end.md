# End-to-end: two synthetic sessions

This example constructs a complete temporary experiment, declares its catalog,
loads both sessions, derives a trial table, and slices the continuous signal for
one trial. It exercises the same public components used by a real workflow
without requiring laboratory data.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from data_conduit.datastructures import (
    DataStructure,
    StreamCatalog,
    TrialSpec,
    first_matching,
    parse_trials,
    slice_stream_for_trial,
)


# ----- Experiment policy: what one trial means ------------------------------

def trial_fields(window: pd.DataFrame, closing_row: pd.Series) -> dict:
    return {
        "cue_time": first_matching(window, "Cue shown"),
        "response_time": first_matching(window, "Response detected"),
        "outcome": closing_row["Event"].split(": ", maxsplit=1)[-1],
    }


trial_spec = TrialSpec(
    closes_trial="Trial ended",
    session_start="Session started",
    fields=trial_fields,
    segments={"response": ("cue_time", "response_time")},
)


# ----- Per-session readers --------------------------------------------------

def read_events(session_path: Path) -> pd.DataFrame:
    return pd.read_csv(session_path / "events.csv", index_col="Time")


def read_signal(session_path: Path) -> pd.DataFrame:
    return pd.read_csv(session_path / "signal.csv", index_col="Time")


# ----- Cross-object configuration -------------------------------------------

def build_trials(objects: dict[str, object]) -> dict[str, object]:
    configured = dict(objects)
    events = configured.pop("events")
    configured["trials"] = parse_trials(events, trial_spec)
    return configured


catalog = StreamCatalog()
catalog.add_reader("events", read_events)
catalog.add_reader("signal", read_signal)
catalog.add_configurator("build_trials", build_trials)


# ----- A temporary two-session experiment ----------------------------------

with TemporaryDirectory() as directory:
    root = Path(directory)

    event_names = [
        "Session started",
        "Cue shown",
        "Response detected",
        "Trial ended: success",
        "Cue shown",
        "Response detected",
        "Trial ended: failure",
    ]
    event_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.5, 6.0]

    for session_name, offset in [("session-a", 0.0), ("session-b", 1.0)]:
        session_path = root / session_name
        session_path.mkdir()

        pd.DataFrame(
            {"Event": event_names},
            index=pd.Index(event_times, name="Time"),
        ).to_csv(session_path / "events.csv")

        signal_times = np.arange(0.0, 6.5, 0.5)
        pd.DataFrame(
            {"value": np.sin(signal_times) + offset},
            index=pd.Index(signal_times, name="Time"),
        ).to_csv(session_path / "signal.csv")

    # Sessions are immediate children of root, so depth=0.
    workflow = DataStructure(root, catalog, depth=0)

    print("selected:", list(workflow.select()))
    streams = workflow.load()
    print("streams:", list(streams))

    trials = streams["trials"]
    signal = streams["signal"]

    first_success = trials.loc[trials["outcome"] == "success"].iloc[0]
    response_signal = slice_stream_for_trial(
        signal,
        first_success,
        segment="response",
        spec=trial_spec,
    )

    print(trials)
    print(response_signal)
```

## What happened

1. `DataStructure.select()` found the two immediate session folders.
2. The catalog read `events.csv` and `signal.csv` independently in each session.
3. The configurator consumed each raw event log and produced a trial table.
4. The catalog normalised both per-session outputs into supported streams.
5. `DataStructure.load()` combined like-named streams and added `session`
   provenance.
6. The slicing helper used the chosen trial's session and response bounds to
   select the matching portion of the continuous signal.

## Adapt the pattern

For a real experiment, keep the orchestration shape and replace only the parts
that encode local knowledge:

- point the readers at the actual files or datasource/integration objects;
- set `depth`, `level_names`, and selectors for the directory hierarchy;
- define a `TrialSpec` using the experiment's event vocabulary;
- add configurators for clock conversion, DLC/video alignment, or derived data;
- mark a reader optional only when that source may legitimately be absent.

Calling `select()` before `load()` provides a read-only view of the intended
session scope. After loading, `workflow.sessions`, `workflow.containers`, and
`workflow.data` expose the three orchestration stages for inspection.

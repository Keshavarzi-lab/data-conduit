# Parse trials and slice another stream

`TrialSpec` separates an experiment's event vocabulary from the generic parsing
algorithm. It declares what closes a trial, how the first trial starts, which
fields should be extracted, and which named time segments should be emitted.

## Describe the experiment

```python
import pandas as pd

from data_conduit.datastructures import (
    TrialSpec,
    first_matching,
    parse_trials,
    slice_stream_for_trial,
    slice_stream_per_trial,
)

events = pd.DataFrame(
    {
        "Event": [
            "Session started",
            "Cue shown",
            "Response detected",
            "Trial ended: success",
            "Cue shown",
            "Response detected",
            "Trial ended: failure",
        ]
    },
    index=pd.Index([0.0, 1.0, 2.0, 3.0, 4.0, 5.5, 6.0], name="Time"),
)


def trial_fields(window: pd.DataFrame, closing_row: pd.Series) -> dict:
    return {
        "cue_time": first_matching(window, "Cue shown"),
        "response_time": first_matching(window, "Response detected"),
        "outcome": closing_row["Event"].split(": ", maxsplit=1)[-1],
    }


def derived_fields(row: dict) -> dict:
    return {
        "response_latency": row["response_time"] - row["cue_time"],
    }


spec = TrialSpec(
    closes_trial="Trial ended",
    session_start="Session started",
    fields=trial_fields,
    derived=derived_fields,
    segments={
        "response": ("cue_time", "response_time"),
    },
)

trials = parse_trials(events, spec)
print(trials)
```

Every event whose text starts with `closes_trial` produces one row. Named
segments become standard columns—in this case `response_start_time` and
`response_end_time`. The whole trial is always available through `start_time`
and `end_time`.

Trial windows are inclusive at both ends. With `start_buffer=0`, the next trial
begins at the preceding trial's closing timestamp; events sharing that boundary
remain available to the parser.

## Slice a combined stream

Cross-session slicing uses both source-session identity and time. Add the
session identity here to mirror a trial table returned from `DataStructure`:

```python
trials["session"] = "session-a"

signal = pd.DataFrame(
    {
        "value": [10, 11, 12, 13, 14, 15, 16],
        "session": ["session-a"] * 7,
    },
    index=pd.Index([0.0, 1.0, 2.0, 3.0, 4.0, 5.5, 6.0], name="Time"),
)

first_response = slice_stream_for_trial(
    signal,
    trials.iloc[0],
    segment="response",
    spec=spec,
)

print(first_response)
```

Supplying `spec` validates the segment name and resolves its emitted bound
columns. Without a spec, the slicer uses the conventional
`{segment}_start_time` and `{segment}_end_time` names.

## Slice several selected trials

Scientific selection remains a normal pandas operation. Filter first, then ask
for one stream slice per retained row:

```python
successful = trials.loc[trials["outcome"] == "success"]

successful_responses = slice_stream_per_trial(
    signal,
    successful,
    segment="response",
    spec=spec,
)
```

The returned list preserves trial-row order. Each item retains the original
pandas/xarray type and contains data only from that trial's source session and
time interval.

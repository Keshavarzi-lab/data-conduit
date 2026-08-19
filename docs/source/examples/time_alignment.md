# Align clocks and build an index map

Independent acquisition systems usually record the same TTL pulse train on
different clocks. `data-conduit` fits a linear target-to-reference conversion
from paired pulse edges, then provides separate tools for mapping arbitrary
stream timestamps onto a shared clock.

## Fit a TTL conversion

The example uses synthetic rising and falling edges. The target clock has a
small rate difference and a different origin.

```python
import numpy as np

from data_conduit.core.sync import (
    build_pulse_table,
    get_ttl_timebase_conversion,
)

reference_starts = np.array([100.0, 110.0, 120.0, 130.0])
target_starts = np.array([5.0, 15.01, 25.02, 35.03])

reference_pulses = build_pulse_table(
    reference_starts,
    reference_starts + 0.2,
)
target_pulses = build_pulse_table(
    target_starts,
    target_starts + 0.2,
)

converted_pulses, ratio, model, fit = get_ttl_timebase_conversion(
    reference_pulses,
    target_pulses,
    use="start",
    normalise_start=False,
    return_details=True,
)

print(fit)
print(converted_pulses[["Start_to_ref", "End_to_ref"]])
```

The fitted model follows:

```text
reference_time ≈ slope × target_time + intercept
```

Use the returned `TTLSyncModel` to transform timestamps from the same target
clock:

```python
target_events = np.array([7.5, 18.0, 29.5])
events_in_reference_time = model.transform(target_events)
```

At least two aligned pulses are required. Pulse tables are paired by ordinal
position and truncated to the shorter length, so inspect the pulse trains and
fit quality before applying a model to experimental data.

## Create a shared global clock

Clock conversion and global indexing solve different problems. Conversion puts
values in the same units/domain; a global clock supplies regular query points.

```python
from data_conduit.core.globaltimes import create_global_clock, index_map_util

global_clock = create_global_clock(
    start_time=100.0,
    end_time=130.0,
    timestep_interval=5.0,
)

stream_times = np.array([100.1, 104.9, 110.2, 119.8, 125.1, 129.9])

index_map = index_map_util(
    global_clock,
    stream_times,
    match_type="nearest",
)

print(index_map["full"])
```

The full table contains the global index/time, matched stream index/time, and
their delta. Available matching rules are `nearest`, `before`, `after`, and
`exact`. Stream times must be monotonically non-decreasing.

## Extract one global-time window

`get_segment` can return values from an index table whose global times fall
inside an inclusive window:

```python
from data_conduit.core.segment import get_segment

stream_times_for_window = get_segment(
    data=index_map["full"],
    start=105.0,
    end=120.0,
    lookup_column="global_time",
    return_column="stream_time",
)
```

Those matched values can then drive `.sel()`, `.loc[]`, or another lookup on the
corresponding stream. For streams already combined by `DataStructure`, prefer
the session-aware trial slicing helpers so equal timestamps from different
sessions cannot be mixed.

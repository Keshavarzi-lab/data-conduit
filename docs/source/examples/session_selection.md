# Select sessions and retain directory levels

Session selection is independent from data loading. Use it to describe the
directory layout and inspect exactly which folders a workflow would process.

For a root containing `mouse/day/session`, there are two intermediate levels, so
`depth=2` and the levels can be named `mouse` and `day`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from data_conduit.core.utils import starts_with
from data_conduit.datastructures import select_sessions

with TemporaryDirectory() as directory:
    root = Path(directory)

    for relative in [
        ("mouse-a", "day-01", "session-001"),
        ("mouse-a", "day-02", "session-002"),
        ("mouse-b", "day-01", "session-001"),
    ]:
        root.joinpath(*relative).mkdir(parents=True)

    sessions = select_sessions(
        root,
        depth=2,
        level_names=("mouse", "day"),
        l0_selector=starts_with("mouse-"),
        l1_selector="day-01",
    )

    for session_id, session in sessions.items():
        print(session_id)
        print("  path:", session.path)
        print("  levels:", session.levels)
        print("  metadata:", session.metadata)
```

`SessionRef.levels` contains identity/grouping information derived from the
directory hierarchy. When streams are later combined, these explicit levels are
broadcast onto the resulting rows or time points.

`SessionRef.metadata` is separate. It contains values produced by metadata
extractors and is retained for inspection, but it is not automatically copied
onto every combined element.

## Selector positions

Selectors target intermediate levels using zero-based positions:

| Directory position | Name in this example | Keyword |
| --- | --- | --- |
| First level below root | `mouse` | `l0_selector` |
| Second level below root | `day` | `l1_selector` |
| Session folder | session name | `include` or `exclude_names` |

Each level selector accepts:

- a string for one exact name;
- a list or tuple for several names;
- a callable receiving the candidate name;
- `None` to accept everything.

Helpers in `data_conduit.core.utils` create common callable selectors:

```python
from data_conduit.core.utils import contains, ends_with, exclude, starts_with

only_mice = starts_with("mouse-")
only_first_days = ends_with("01")
without_pilot = exclude("pilot")
has_training = contains("training")
```

## Include or exclude sessions

The leaf-level modes are mutually exclusive:

```python
from pathlib import Path

from data_conduit.datastructures import select_sessions


def compare_leaf_filters(root: Path):
    chosen = select_sessions(
        root,
        depth=2,
        level_names=("mouse", "day"),
        include=["session-001", "session-002"],
    )

    remaining = select_sessions(
        root,
        depth=2,
        level_names=("mouse", "day"),
        exclude_names=["session-002"],
    )
    return chosen, remaining
```

When the same session-folder name occurs in more than one branch, the returned
mapping uses a relative path as the identifier so entries do not overwrite each
other. That identifier becomes the default cross-session provenance value.

`DataStructure.select()` applies these same rules without running any readers,
which makes it the safest inspection step before a full load.

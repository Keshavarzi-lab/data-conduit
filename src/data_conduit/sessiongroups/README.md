# `sessiongroups`

Select multiple experimental sessions from a data directory, load each one
into a `Session`, optionally **align** every structure onto one shared
zero-based timebase, then **combine** a data-structure type across sessions
into a single multi-session structure you can filter and query.

```
select_sessions  ->  build_sessions  ->  combine_sessions
   (select +              (load each session +        (multi-session
    label)                 optionally align)           "database")
```

Alignment is a **pipeline of small functions** (in `alignment.py`), not a
class. Each step is optional; the caller composes only what they need.

---

## 1. Select sessions: `select_sessions`

Walk a data directory and choose session folders. Three filter modes, and a
**configurable depth** for where sessions live relative to the root:

```python
from data_conduit.sessiongroups import select_sessions

# Q_C layout: mouse / phase / day / <session>. Sessions are 3 levels down.
sel = select_sessions(
    'path/to/data',
    depth=3,
    level_names=('mouseID', 'phase', 'day'),   # names for the intermediate levels
    l1_selector='Testing',                      # optional filter on an intermediate level
    # ALL:      (pass neither include nor exclude_names)
    # INCLUDE:  include=['2026-04-30T170040Z', ...]   -> only these
    # EXCLUDE:  exclude_names=['FLR_M01569521']       -> all except these
)
# -> {session_name: {'path': Path, 'mouseID': ..., 'phase': ..., 'day': ..., 'label': ...}}
```

### Labelling: extractors

By default each session is labelled with its **folder name as a string**
(`NameExtractor`). To derive a richer label, pass an extractor:

```python
from data_conduit.sessiongroups import DateTimeExtractor

sel = select_sessions(
    'path/to/data', depth=3,
    extractors=DateTimeExtractor(),   # parse the name -> datetime (key 'datetime')
)
```

`DateTimeExtractor` auto-detects common formats, or takes `formats=` /
`convention=` and a `pattern=` regex. Subclass `LabelExtractor` to add new
label sources.

---

## 2. Choose what to extract: `DataStructureCatalog`

A **catalog** lists which data structures to load per session. Start from
the lab defaults and tweak it (toggle, add, or remove) without touching
the loader:

```python
from data_conduit.sessiongroups import default_harp_catalog, DataStructureSpec

catalog = default_harp_catalog('device.yml', 'soundcard.yml')
# events (required), nosepoke, soundcard, camera, video, session_settings

catalog.disable('video')            # temporarily leave one out
catalog.remove('camera')            # drop one entirely
catalog.add(DataStructureSpec(      # add your own
    name='rotation',
    reader=lambda p: RotationData(experiment_directory_path=p).df,
))
```

Each `DataStructureSpec` has `name`, `reader` (a
`path -> source / DataArray / DataFrame` callable), `enabled`, `required`
(if `False`, a missing structure is skipped), and `sync` (set for
cross-clock data; see §4).

---

## 3. Build + combine: `build_sessions`, `combine_sessions`

```python
from data_conduit.sessiongroups import build_sessions, combine_sessions

# Load every selected session; returns an ordered SessionGroup.
group = build_sessions(
    sel,
    catalog,
    normalise=True,                      # opt-in: shift each session to t=0
    global_clock={'timestep': 1.0},      # opt-in: build regular clock + index tables per session
)

# Stack one structure TYPE across sessions into a single multi-session object.
nosepoke = combine_sessions(group, dim='Time', type_name='nosepoke:Activations')
# -> one xr.DataArray spanning all sessions, with a 'label' coordinate marking
#    which session each sample came from. Sorted by session, then by time.
```

The combined object keeps its **native type**. A Nosepoke stays an
`xr.DataArray` (so `.ulookup.select(device='Behavior0')` still works); a
trial table stays a long `DataFrame` with a `label` column. Omit
`type_name` to get a combined `Session` holding every member.

### Tailoring a combined table

```python
from data_conduit.sessiongroups import add_label_column, drop_columns

nosepoke = add_label_column(
    nosepoke, 'is_first',
    lambda da: da['label'].values == first_label, dim='Time',
)
nosepoke = drop_columns(nosepoke, ['register'])
```

---

## 4. Per-session alignment pipeline

`build_sessions` calls these under the hood, but you can use any of them
directly on a single session:

```python
from data_conduit.sessiongroups import (
    load_session,
    normalise_to_zero,
    attach_cross_clock,
    build_global_clock,
    build_index_tables,
)

# 1) Load same-clock structures into a plain Session (no time math).
session = load_session('path/to/session', catalog)

# 2) Optionally shift everything so the earliest log sits at t=0.
session = normalise_to_zero(session)             # t0_from='session' by default

# 3) Optionally bring in cross-clock structures (Neuropixels, DLC) via TTL.
session, ttl_models = attach_cross_clock(session, catalog)

# 4) Optionally build a regular global clock and per-member index tables.
clock = build_global_clock(session, timestep=1.0)
tables = build_index_tables(session, clock)

session.data            # bundle: {'events': df, 'nosepoke:Activations': DataArray, ...}
session.metadata['t0']  # the session's earliest time, if normalised
```

### Cross-clock data (Neuropixels / DLC)

HARP / Bonsai structures share one clock. Data on a different clock
(Neuropixels, DLC pose) is handled by giving its spec a `sync` config and
a reader that returns `{'pulse_table', 'data', 'time_column'}`;
`attach_cross_clock` fits a TTL conversion (via `data_conduit.sync.ttl`)
onto the reference clock. These readers are **caller-supplied** (e.g.
`movement` for DLC, pre-extracted NPX sync / spike times). The package
depends on neither.

---

## Building blocks

`select_sessions` / `build_sessions` / `combine_sessions` are thin
orchestration over the lower layers, all of which remain available:

- `bundle` primitives: `map_bundle`, `select_bundle`,
  `select_bundle_where`, `combine_bundles`, `split_bundle`
  (alignment-preserving ops over a bundle of `DataArray` / `Dataset` /
  `DataFrame` members).
- `Session` / `SessionGroup`: immutable, chainable wrappers. Every op
  returns a new object; `combine` / `split` round-trip.

See [`alignment.py`](alignment.py), [`selection.py`](selection.py),
[`data_structures.py`](data_structures.py), [`database.py`](database.py),
and [`bundle.py`](bundle.py) for full docstrings.

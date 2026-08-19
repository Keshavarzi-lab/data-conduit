# Overview

`data-conduit` turns heterogeneous experimental files into named, reusable data
streams. It is designed for sessions that contain several acquisition systems,
file formats, clocks, and directory levels, but its central abstractions are
ordinary Python callables, pandas objects, and xarray objects.

The library does not prescribe one laboratory directory convention. Instead,
you describe where sessions are located, how each source should be read, and
which transformations should run before streams are combined.

## What the package provides

| Capability | Main entry points | Result |
| --- | --- | --- |
| Discover and read files | `data_conduit.core.io` | A nested dictionary mirroring the directory tree |
| Wrap one source | `data_conduit.datasources.monosource` | Named pandas/xarray outputs from one location |
| Combine related sources | `data_conduit.datasources.multisource` | Unified arrays with queryable virtual coordinates |
| Select and combine sessions | `data_conduit.datastructures` | Named streams combined across the selected sessions |
| Align independent clocks | `data_conduit.core.sync` and `data_conduit.core.globaltimes` | Fitted time conversion and shared index maps |
| Parse and slice trials | `data_conduit.datastructures` | Trial tables and session-aware stream slices |
| Load supported ecosystems | `data_conduit.integrations` | HARP and DeepLabCut adapters |

## Two complementary workflows

### Work with one source directly

Use the core I/O functions or a datasource when you need one session or one
kind of file. `MonoSource` turns selected leaves from a nested file tree into
named xarray objects. `MultiSource` combines values that are physically spread
across several devices, registers, or channels and attaches virtual coordinates
such as `device` and `register`.

```python
from data_conduit.core.io import collect_dfs, read_csv

tables = collect_dfs(
    "path/to/session",
    readers={".csv": read_csv},
)
```

### Describe a reusable multi-session pipeline

Use `StreamCatalog` and `DataStructure` when the same loading recipe should be
applied to many sessions. In this schematic, `read_events` and `build_trials`
are the reader and configurator functions supplied by the workflow:

1. A `DataStructure` selects session directories.
2. Its `StreamCatalog` runs named readers for each session.
3. Ordered configurators align, derive, replace, or remove loaded objects.
4. Objects are normalised into pandas/xarray streams.
5. Like-named streams are combined across sessions with their provenance.

```python
from data_conduit.datastructures import DataStructure, StreamCatalog

catalog = StreamCatalog()
catalog.add_reader("events", read_events)
catalog.add_configurator("trials", build_trials)

pipeline = DataStructure(
    "path/to/experiment",
    catalog,
    depth=2,
    level_names=("mouse", "day"),
)

streams = pipeline.load()
```

Readers and configurators remain normal functions. A reader accepts one session
directory and returns an object. A configurator accepts the complete mapping of
objects loaded for that session and returns the revised mapping. This keeps
dependencies between streams explicit and makes a laboratory-specific workflow
possible without changing the library.

## The common stream boundary

The orchestration layer combines three analysis-ready types:

- `pandas.DataFrame`
- `xarray.DataArray`
- `xarray.Dataset`

Reader results may also be lightweight wrappers exposing `.df` or
`.data_arrays`, or mappings containing the supported types. The catalog
normalises these forms before cross-session combination. Each combined value is
tagged with its source `session`; named directory levels such as `mouse` and
`day` are attached as additional provenance.

## Optional sources are explicit

A catalog reader is required by default. Mark a reader as optional only when a
missing file is legitimate for some sessions:

```python
catalog.add_reader("pose", read_pose, optional=True)  # workflow-supplied reader
```

Only `FileNotFoundError` is treated as an absent optional source. Invalid data
and programming errors still propagate, so a partially populated experiment
does not hide unrelated failures.

## Integrations

The HARP integration supplies device-aware readers and datasource presets. The
DeepLabCut integration reads frame-indexed pose and can attach camera-frame
timestamps so pose shares the session clock. These integrations feed the same
catalog and stream abstractions as custom readers.

Start with the {doc}`workflow guide <workflow>` for the full component model,
or open the {doc}`example gallery <examples>` for focused recipes.

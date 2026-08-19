# data-conduit

A Python toolkit for constructing reusable data-processing workflows for
multi-stream experimental data.

> **Status: alpha.** The public API and documentation are still evolving.

[Documentation](https://keshavarzi-lab.github.io/data-conduit/) ·
[Repository](https://github.com/Keshavarzi-lab/data-conduit) ·
[Issue tracker](https://github.com/Keshavarzi-lab/data-conduit/issues)

![The data-conduit workflow](https://raw.githubusercontent.com/Keshavarzi-lab/data-conduit/main/docs/source/_static/images/data-flow.svg)

## What it provides

data-conduit separates data discovery, source-specific loading, session
organisation, and cross-session combination:

- filesystem discovery and tabular readers;
- reusable mono-source and multi-source data loaders;
- explicit session selection and metadata extraction;
- configurable per-session readers and cross-source transforms;
- named pandas and xarray streams combined across sessions;
- timestamp collection, clock alignment, TTL synchronisation, segmentation,
  and virtual-coordinate lookup;
- optional HARP and DeepLabCut integrations.

The core package depends only on NumPy, pandas, xarray, and PyYAML. Hardware,
pose, notebook, and plotting dependencies are provided through optional extras.

## Data flow

The high-level orchestration API is in <code>data_conduit.datastructures</code>:

1. <code>select_sessions</code> walks an experiment hierarchy and returns a
   <code>dict[str, SessionRef]</code>. Each reference keeps the session path,
   explicit hierarchy levels, and separately extracted metadata.
2. One <code>StreamCatalog</code> is applied independently to every selected
   <code>SessionRef.path</code>. Its readers load source objects, then its
   ordered configurators can align, derive, replace, or remove objects using
   the full per-session object mapping.
3. Each configured session is normalised to a per-session
   <code>StreamMap</code>: <code>dict[str, DataObject]</code>, where a data
   object is a pandas <code>DataFrame</code> or an xarray
   <code>DataArray</code> or <code>Dataset</code>.
4. Like-named streams from different sessions are regrouped into
   <code>StreamContainer</code> objects.
5. Each container combines its members and attaches the source session and
   declared <code>SessionRef.levels</code>. The resulting cross-session
   streams form the final <code>StreamMap</code> exposed by a loaded
   <code>DataStructure</code>.

Session metadata remains available on its <code>SessionRef</code>; it is not
broadcast onto every row or time point unless it was explicitly declared as a
hierarchy level.

## Installation

From a source checkout, install the core package with:

~~~bash
python -m pip install .
~~~

Install one or more optional feature groups when needed:

~~~bash
python -m pip install ".[harp]"
python -m pip install ".[pose,movement]"
python -m pip install ".[notebooks]"
python -m pip install ".[all]"
~~~

| Extra | Adds |
|---|---|
| <code>harp</code> | HARP register decoding through <code>harp-python</code> |
| <code>pose</code> | PyTables support for DeepLabCut HDF5 output |
| <code>movement</code> | Interoperability with the <code>movement</code> package |
| <code>notebooks</code> | Jupyter, interactive tables, and plotting |
| <code>all</code> | All optional feature groups |

## Quick example

<code>collect_dfs</code> walks a directory tree and applies readers according
to file extension:

~~~python
from data_conduit.core.io import collect_dfs, read_csv

data = collect_dfs(
    base_path="path/to/session",
    readers={".csv": read_csv},
)
~~~

The returned nested dictionary mirrors the discovered directory structure.
Selectors such as <code>l0_selector</code> can be supplied when only part of
that hierarchy should be loaded.

## Package layout

~~~text
data_conduit
├── core
│   ├── datetime
│   ├── globaltimes
│   ├── io
│   ├── segment
│   ├── sync
│   ├── timestamps
│   ├── utils
│   └── virtualarrays
├── datasources
│   ├── monosource
│   └── multisource
├── datastructures
├── integrations
│   ├── DLC
│   └── harp
├── validators
└── qc
~~~

- <code>data_conduit.core</code> contains format-agnostic I/O, time,
  alignment, segmentation, and array utilities.
- <code>data_conduit.datasources</code> contains reusable source loaders.
- <code>data_conduit.datastructures</code> owns session selection, catalogues,
  stream normalisation, combination, trial parsing, and slicing.
- <code>data_conduit.integrations</code> contains optional
  ecosystem-specific adapters.
- <code>data_conduit.validators</code> contains shared input validation.
- <code>data_conduit.qc</code> contains experiment-specific QC helpers; its
  notebooks and generated media are repository resources rather than runtime
  package data.

## Documentation

Documentation site: [GitHub Pages](https://keshavarzi-lab.github.io/data-conduit/).

To check the generated API page and build the documentation locally:

~~~bash
python -m pip install -r docs/requirements.txt
make docs-api-check
make docs
~~~

The Sphinx output is written below <code>docs/build/html/</code>.

## Development

~~~bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
~~~

Bug reports and design discussions are welcome in the
[issue tracker](https://github.com/Keshavarzi-lab/data-conduit/issues).
Contributions can be proposed through
[pull requests](https://github.com/Keshavarzi-lab/data-conduit/pulls).

## License

data-conduit is distributed under the [MIT License](LICENSE).

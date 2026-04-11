# `data-conduit` — API Reference

**Auto-generated from source docstrings** — reflects the current state of the library.

| Section | What it covers |
|---------|----------------|
| [1. Core Workflow Modules](#1-core-workflow-modules) | IO, MonoSource, HARP Presets, MultiSource, Virtual Arrays, Segmentation |
| [2. Synchronisation & Alignment](#2-synchronisation--alignment) | TTL Sync, Global Times |
| [3. Internals & Utilities](#3-internals--utilities) | Timestamps, HarpTools, Utils, Validators |

---
## 1. Core Workflow Modules


### IO — `data_conduit.io`
```python
import data_conduit.io
```

The main entry point for loading data. `collect_dfs` walks a directory tree and returns a nested dictionary of DataFrames, with file readers dispatched by extension. The reader registry (`add_reader` / `get_reader`) lets you plug in custom formats alongside the built-in CSV, JSON, JSONL, and YAML readers.

---

#### `add_reader(name: str, reader_fn: <built-in function callable>) -> None`
> **func**

Add a new reader function to the registry with a specified name. Parameters: name (str): The name to register the reader function under (e.g., 'csv', 'json'). Note: this name is used to reference the reader when calling "collect_dfs". Name should be unique and descriptive of the file format or reading method. reader_fn (callable): The reader function to register. Must follow the signature: (file_path: str | Path, **kwargs) -> pd.DataFrame Returns: None

---

#### `collect_dfs(...)`
> **func**

Collect files from a directory tree into a nested dictionary of DataFrames. Walks the entire directory tree from base_path, building a nested dict that mirrors the folder structure. Files with extensions matching a provided reader are read into DataFrames. Files without a matching reader are skipped. If no readers are provided, all files are stored as Paths. Level selectors filter at any depth during the walk, supporting exact match lists, single strings, callables, and None (wildcard). Wrappers can map human-readable arguments to level selectors internally.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_path` | `str or Path` | Root directory to walk. |
| `readers` | `dict[str, Callable] or None` | Mapping of file extensions to reader functions. Each reader must follow: (file_path, **kwargs) -> pd.DataFrame. e.g. {'.bin': read_harp_bin, '.csv': read_csv} If None, all files are stored as Paths. |
| `reader_kwargs` | `dict[str, dict] or None` | Mapping of file extensions to kwargs dicts for each reader. e.g. {'.bin': {'harp_reader': r, 'addr_to_name': m}} Extensions not in this dict receive no extra kwargs. |
| `keep_empty` | `bool` | If True, preserve empty subdirectories as empty dicts. Default is True. |
| `flatten` | `bool` | If True, return a flat dict with keys joined by separator. |
| `separator` | `str` | Separator for flattened keys. Default is ':'. |
| `verbose` | `bool` | If True, prints warnings during processing. |
</details>

**Returns:** `dict` — Nested or flat dictionary of DataFrames (or Paths if no readers).

---

#### `collect_file_paths(...)`
> **func**

Walk a directory tree and build a nested dictionary of file Paths. Directory names become dictionary keys at each level. Files matching the pattern become leaf values, keyed by their stem (filename without extension).

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_path` | `str or Path` | The root directory to walk. |
| `file_pattern` | `str` | Glob pattern for matching files (e.g. '*.csv', '*.bin', '*'). Only applied to files, not directories. |
| `depth` | `int or None` | Maximum number of directory levels to descend into. None means unlimited depth. 0 means only files directly in base_path (no subdirectories). 1 means files in base_path and one level of subdirectories, etc. |
| `_current_depth` | `int` | Internal recursion tracker. Do not set manually. |
</details>

**Returns:** `dict` — A nested dictionary where: - Directory names are keys leading to nested dicts - File stems are keys leading to Path values

---

#### `collect_folders(...)`
> **func**

Collect folders from a base path that match a specified prefix. Only collects folders one level deep (i.e., does not search subdirectories). Only collects directories, not files.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_path` | `str \| Path` | The base directory to search for folders. |
| `folder_prefix` | `str \| None` | A prefix to filter folders. Only folders starting with this prefix will be collected. If None, all folders in the base path are collected. |
</details>

**Returns:** `list[Path]` — A list of Path objects representing the collected folders.

---

#### `get_reader(name: str) -> <built-in function callable>`
> **func**

Retrieve a reader function from the registry by name. Parameters: name (str): The name of the reader function to retrieve (e.g., 'csv', 'json'). Returns: callable: The reader function registered under the specified name. Raises: KeyError: If no reader is registered under the specified name.

---

#### `read_csv(path: str | pathlib.Path, **kwargs) -> pandas.core.frame.DataFrame`
> **func**

Read a CSV file into a pandas DataFrame. Uses following as default kwargs if no kwargs are provided: - header=0, dtype=str, engine='python' - sep splits on commas not inside braces - first column as index Parameters path : str or Path Path to the CSV file. **kwargs Keyword arguments passed directly to pd.read_csv. If no kwargs are provided, defaults are used. Returns pd.DataFrame The loaded DataFrame.

---

#### `read_files(...)`
> **func**

Walk a nested dictionary of file Paths and apply a reader function to each, producing a nested dictionary of DataFrames with the same structure.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `paths_dict` | `dict` | A nested dictionary where leaf values are Path objects (as produced by collect_file_paths). |
| `reader_fn` | `callable or None` | A function with signature (path: Path, **kwargs) -> pd.DataFrame. If None, leaves are returned as-is (passthrough mode). |
| `reader_kwargs` | `dict or None` | Extra keyword arguments passed to reader_fn for every file. |
| `verbose` | `bool` | If True, prints warnings when a reader fails on a file. |
</details>

**Returns:** `dict` — A nested dictionary with the same key structure as paths_dict, but with DataFrames (or reader output) as leaf values.

---

#### `read_json(path: str | pathlib.Path, **kwargs) -> pandas.core.frame.DataFrame`
> **func**

Read a JSON or JSONL file into a pandas DataFrame. Uses sensible defaults if no kwargs are provided: - lines=True (JSONL format) - dtype=str, orient='records' Parameters path : str or Path Path to the JSON/JSONL file. **kwargs Keyword arguments passed directly to pd.read_json. If no kwargs are provided, defaults are used. Returns pd.DataFrame The loaded DataFrame.

---

#### `split_jsonl(...)`
> **func**

Load a JSONL file and split into metadata and trials DataFrames. Each JSONL record produces one metadata row and multiple trial rows.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str or Path` | Path to the JSONL file. |
| `verbose` | `bool` | If True, prints detailed output during loading. |
</details>

**Returns:** `tuple[pd.DataFrame, pd.DataFrame]` — A tuple of (metadata_df, trials_df).

---

#### `split_yaml(...)`
> **func**

Load a YAML file and split into metadata and trials DataFrames.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str or Path` | Path to the YAML file. |
| `verbose` | `bool` | If True, prints detailed output during loading. |
</details>

**Returns:** `tuple[pd.DataFrame, pd.DataFrame]` — A tuple of (metadata_df, trials_df).

---

### MonoSource — `data_conduit.datasources.monosource`
```python
import data_conduit.datasources.monosource
```

Convenience wrappers around `collect_dfs` that preconfigure readers, level selectors, and named DataArray extraction for common experimental data types. `MonoSource` is the base class; `FileTypeData` handles flat file formats. Presets like `ExperimentEvents`, `VideoData`, and `RotationData` provide one-line loading for standard Bonsai workflow outputs.

---

#### `ExperimentEvents(...)`
> **class**

Preset config for ExperimentEvents CSV data. Loads CSV files from ExperimentEvents subfolder(s), renames the 'Value' column to 'Event', and ensures the index is named 'Time'.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_directory_path` | `str or Path or None` | Path to the experiment directory. |
| `device_type` | `str` | Folder prefix. Default 'ExperimentEvents'. |
| `reader_kwargs` | `dict or None` | Custom kwargs for read_csv. |
| `rename_columns_dict` | `dict` | Column renaming. Default {'Value': 'Event'}. |
| `rename_index_dict` | `str` | Index name. Default 'Time'. |
| `filetype_data_arrays` | `dict` | Mapping of friendly names to level selectors. Default: {'events': {'l0_selector': 'ExperimentEvents'}} |
| `verbose` | `bool` | If True, print warnings during processing. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from filetype_data_arrays. Default ExperimentEvents data_arrays: { 'events': xr.DataArray for ExperimentEvents CSV data, } |
| `df` | `pd.DataFrame` | Backward-compatible access to the single loaded DataFrame. |
</details>

---

#### `FileTypeData(...)`
> **class**

MonoSource for common file-type data (CSV, JSON, JSONL, YAML). Extends MonoSource to simplify loading directories of flat files. Analogous to Device: maps a file_type string to the appropriate reader, uses device_type to filter folders via l0_selector, and applies post-load column/index renaming.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `dfs_dict` | `dict or None` | Optional pre-built nested dictionary of DataFrames. If provided, collect_dfs is not called and this dict is used directly. |
| `experiment_directory_path` | `str or Path` | Root directory to walk. |
| `device_type` | `str or None` | Folder prefix to filter on (e.g. 'ExperimentEvents', 'VideoData'). Maps to l0_selector=starts_with(device_type). If None, no folder filtering is applied. |
| `file_type` | `str` | File format to load. Determines which reader and extension to use. Supported: 'csv', 'json', 'jsonl', 'yml', 'yaml'. |
| `reader_kwargs` | `dict or None` | Kwargs passed to the reader function for every matching file. e.g. {'header': 0, 'index_col': 0} for CSV. If None, the reader uses its built-in defaults. |
| `rename_columns_dict` | `dict or None` | Column renaming applied to all leaf DataFrames after loading. e.g. {'Value': 'Event'} |
| `rename_index_dict` | `str or dict or None` | Index renaming applied to all leaf DataFrames after loading. If str, renames the index to that name (e.g. 'Time'). If dict, passed to df.index.rename(). |
| `filetype_data_arrays` | `dict or None` | Mapping of friendly names to level selectors for extracting named DataArrays from dfs_dict. Uses the same l{n}_selector format as collect_dfs. e.g. {'events': {'l0_selector': 'ExperimentEvents'}} |
| `keep_empty` | `bool` | If True, preserve empty subdirectories as empty dicts. |
| `flatten` | `bool` | If True, flatten the output dict. |
| `separator` | `str` | Separator for flattened keys. |
| `verbose` | `bool` | If True, print warnings during processing. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `dfs_dict` | `dict` | Nested dictionary of DataFrames from collect_dfs or provided directly. |
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from filetype_data_arrays. |
| `df` | `pd.DataFrame or dict` | Backward-compatible convenience property. Returns the single leaf DataFrame from dfs_dict when there is only one file. For split types (JSONL/YAML), returns {'metadata': df, 'trials': df}. If multiple leaves exist, returns the full nested dict. |
</details>

---

#### `MonoSource(...)`
> **class**

Base class for data sources in data-conduit. Calls collect_dfs to load data, then optionally builds named DataArrays by navigating dfs_dict using monosource_data_arrays. If dfs_dict is provided, it is used directly and collect_dfs is not called. Subclasses configure what to load by setting readers, reader_kwargs, and level selectors. For example, HarpDevice sets readers={'.bin': read_harp_bin} and FileTypeData sets readers={'.csv': read_csv}.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `dfs_dict` | `dict or None` | Optional pre-built nested dictionary of DataFrames. If provided, collect_dfs is not called and this dict is used directly. |
| `experiment_directory_path` | `str or Path` | Root directory to walk. |
| `readers` | `dict[str, Callable] or None` | Mapping of file extensions to reader functions. e.g. {'.bin': read_harp_bin, '.csv': read_csv} |
| `reader_kwargs` | `dict[str, dict] or None` | Mapping of file extensions to kwargs for each reader. e.g. {'.bin': {'harp_device_yaml_path': 'device.yml'}} |
| `keep_empty` | `bool` | If True, preserve empty subdirectories in dfs_dict as empty dicts. If False, skip them. |
| `flatten` | `bool` | If True, flatten the output dict from collect_dfs. If False, keep the nested structure. e.g. flatten=True turns {'a': {'b': df}} into {'a/b': df}. |
| `separator` | `str` | If flatten=True, the separator to use when joining nested keys. e.g. separator='/' turns {'a': {'b': df}} into {'a/b': df}. |
| `monosource_data_arrays` | `dict or None` | Optional mapping of DataArray names to paths in dfs_dict. If provided, these DataArrays are built from dfs_dict and stored as attributes for easy access. Each value is a tuple of keys navigating the nested dict to find the DataFrame to convert to a DataArray. e.g. {'PlaySoundFreq': ('SoundCard', '32')} for dfs_dict['SoundCard']['32'] -> DataFrame -> DataArray named 'PlaySoundFreq' |
| `verbose` | `bool` | If True, print verbose output during collect_dfs. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `dfs_dict` | `dict` | Nested dictionary of DataFrames from collect_dfs or provided directly. |
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from monosource_data_arrays. |
</details>

---

#### `RingDebugData(...)`
> **class**

Preset config for ring-debug YAML data (metadata + trials). Loads YAML files from the experiment directory (ring-debug.yml typically sits at the root, not in a device subfolder). Each YAML file is split into metadata and trials DataFrames stored as a nested dict: {stem: {'metadata': df, 'trials': df}}.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_directory_path` | `str or Path or None` | Path to the experiment directory. |
| `device_type` | `str` | Folder prefix. Default 'ring-debug'. |
| `rename_columns_dict` | `dict or None` | Column renaming. Default None. |
| `rename_index_dict` | `str` | Index name. Default 'Time'. |
| `filetype_data_arrays` | `dict` | Mapping of friendly names to level selectors. Default: {'ring_debug': {'l0_selector': 'ring-debug'}} |
| `verbose` | `bool` | If True, print warnings during processing. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from filetype_data_arrays. Default RingDebugData data_arrays: { 'ring_debug': xr.DataArray for ring-debug YAML data, } |
| `df` | `dict` | Backward-compatible access. Returns {'metadata': df, 'trials': df}. |
</details>

---

#### `RotationData(...)`
> **class**

Preset config for rotation encoder CSV data. Loads CSV files from a rotation device subfolder (InnerRotation, OuterRotation, or NosepokeRotation), renames 'Value' to 'Rotation', and optionally applies angular unit conversion and range wrapping.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_directory_path` | `str or Path or None` | Path to the experiment directory. |
| `device_type` | `str or None` | Folder prefix. Typically one of 'InnerRotation', 'OuterRotation', or 'NosepokeRotation'. |
| `reader_kwargs` | `dict or None` | Custom kwargs for read_csv. |
| `rename_columns_dict` | `dict` | Column renaming. Default {'Value': 'Rotation'}. |
| `rename_index_dict` | `str` | Index name. Default 'Time'. |
| `angular_unit_conversion` | `str or None` | Unit conversion to apply: 'deg2rad' or 'rad2deg'. If None, no conversion is applied. |
| `angular_range` | `list[float] or None` | Two-element list [min, max] to wrap angular values within. e.g. [0, 360], [-180, 180], [0, 2*pi]. If None, no wrapping is applied. |
| `filetype_data_arrays` | `dict or None` | Mapping of friendly names to level selectors. No default — device_type varies per instantiation. |
| `verbose` | `bool` | If True, print warnings during processing. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from filetype_data_arrays (if provided). |
| `df` | `pd.DataFrame` | Backward-compatible access to the single loaded DataFrame. |
</details>

---

#### `VideoData(...)`
> **class**

Preset config for VideoData CSV data. Loads CSV files from VideoData subfolder(s), renames chunk data columns to friendly names (FrameID, Timestamp).

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_directory_path` | `str or Path or None` | Path to the experiment directory. |
| `device_type` | `str` | Folder prefix. Default 'VideoData'. |
| `reader_kwargs` | `dict or None` | Custom kwargs for read_csv. |
| `rename_columns_dict` | `dict` | Column renaming. Default maps ChunkData fields. |
| `rename_index_dict` | `str` | Index name. Default 'Time'. |
| `filetype_data_arrays` | `dict` | Mapping of friendly names to level selectors. Default: {'video': {'l0_selector': 'VideoData'}} |
| `verbose` | `bool` | If True, print warnings during processing. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from filetype_data_arrays. Default VideoData data_arrays: { 'video': xr.DataArray for VideoData CSV data, } |
| `df` | `pd.DataFrame` | Backward-compatible access to the single loaded DataFrame. |
</details>

---

#### `VisualEnvironment(...)`
> **class**

Preset config for VisualEnvironment CSV data. Loads CSV files from VisualEnvironment subfolder(s) using custom column names for the headerless Bonsai output format.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_directory_path` | `str or Path or None` | Path to the experiment directory. |
| `device_type` | `str` | Folder prefix. Default 'VisualEnvironment'. |
| `reader_kwargs` | `dict` | Custom kwargs for read_csv. Defaults provide column names for the standard VisualEnvironment format. |
| `rename_columns_dict` | `dict or None` | Column renaming. Default None (names set via reader_kwargs). |
| `rename_index_dict` | `str` | Index name. Default 'Time'. |
| `filetype_data_arrays` | `dict` | Mapping of friendly names to level selectors. Default: {'visual_environment': {'l0_selector': 'VisualEnvironment'}} |
| `verbose` | `bool` | If True, print warnings during processing. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from filetype_data_arrays. Default VisualEnvironment data_arrays: { 'visual_environment': xr.DataArray for VisualEnvironment CSV data, } |
| `df` | `pd.DataFrame` | Backward-compatible access to the single loaded DataFrame. |
</details>

---

### HARP Presets — `data_conduit.datasources.presets.harp`
```python
import data_conduit.datasources.presets.harp
```

HARP-specific device and multi-device presets (requires `harp-python`). `Device` handles HARP .bin files; `SoundCard`, `CameraStart`, and `Camera0Frames` are one-line presets for common HARP boards. `MultiDevice` and `Nosepoke` combine data from multiple HARP boards into unified DataArrays with virtual coordinate lookup.

---

#### `Camera0Frames(...)`
> **class**

Preset device wrapper for camera frame registers. In addition to the base Device behavior, this preset can call the matching-time helper to keep only timestamps shared by every loaded DataArray.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_directory_path` | `str or Path or None` | Root path passed to MonoSource when loading from disk. |
| `harp_device_yaml_path` | `str or Path` | YAML definition passed through to ``read_harp_bin``. |
| `device_list` | `list[str] or None` | Expected device names. Used only for warning messages. Camera0Frames Default: ["Behavior0", "Behavior1", "Behavior2", "Behavior3", "Behavior4", "Behavior5"] |
| `device_IDs` | `dict or None` | Placeholder metadata for expected device IDs. Camera0Frames Default: { "Behavior0": "ID_0", "Behavior1": "ID_1", "Behavior2": "ID_2", "Behavior3": "ID_3", "Behavior4": "ID_4", "Behavior5": "ID_5", } |
| `device_registers` | `dict or None` | Expected registers per device. Also used as an optional filter. Camera0Frames Default: { "Behavior0": ["92"], "Behavior1": ["92"], "Behavior2": ["92"], "Behavior3": ["92"], "Behavior4": ["92"], "Behavior5": ["92"], } |
| `device_data_arrays` | `dict or None` | Mapping of friendly names to ``(device, register)`` paths. Camera0Frames Default: { "Behavior0_Camera0Frame": {"l0_selector": "Behavior0", "l1_selector": "92"}, "Behavior1_Camera0Frame": {"l0_selector": "Behavior1", "l1_selector": "92"}, "Behavior2_Camera0Frame": {"l0_selector": "Behavior2", "l1_selector": "92"}, "Behavior3_Camera0Frame": {"l0_selector": "Behavior3", "l1_selector": "92"}, "Behavior4_Camera0Frame": {"l0_selector": "Behavior4", "l1_selector": "92"}, "Behavior5_Camera0Frame": {"l0_selector": "Behavior5", "l1_selector": "92"}, } |
| `rekey` | `bool` | If True, rekey HARP stems to register keys (e.g. convert ``Behavior0_92_1904-01-01T01-00-00.bin`` into ``92``) so the device tree becomes ``device -> register -> dataframe``. Set to False to keep original keys. |
| `matching_only` | `bool` | If True, apply the matching-time helper to keep only timestamps shared by every loaded DataArray. |
| `verbose` | `bool` | If True, print verbose messages during initialisation. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from device_data_arrays. Accessed via device.data_arrays['{friendly_name}'] Default Camera0Frames data_arrays: { "Behavior0_Camera0Frame": xr.DataArray for Behavior0 register 92, "Behavior1_Camera0Frame": xr.DataArray for Behavior1 register 92, "Behavior2_Camera0Frame": xr.DataArray for Behavior2 register 92, "Behavior3_Camera0Frame": xr.DataArray for Behavior3 register 92, "Behavior4_Camera0Frame": xr.DataArray for Behavior4 register 92, "Behavior5_Camera0Frame": xr.DataArray for Behavior5 register 92, } |
</details>

---

#### `CameraStart(...)`
> **class**

Preset device wrapper for camera start registers. This preset provides default behavior devices and exposes the start-camera registers as named device_data_arrays. It adds no extra helper methods.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_directory_path` | `str or Path or None` | Root path passed to MonoSource when loading from disk. |
| `harp_device_yaml_path` | `str or Path` | YAML definition passed through to ``read_harp_bin``. |
| `device_list` | `list[str] or None` | Expected device names. Used only for warning messages. CameraStart Default: ["Behavior0", "Behavior1", "Behavior2", "Behavior3", "Behavior4", "Behavior5"] |
| `device_IDs` | `dict or None` | Placeholder metadata for expected device IDs. CameraStart Default: { "Behavior0": "ID_0", "Behavior1": "ID_1", "Behavior2": "ID_2", "Behavior3": "ID_3", "Behavior4": "ID_4", "Behavior5": "ID_5", } |
| `device_registers` | `dict or None` | Expected registers per device. Also used as an optional filter. CameraStart Default: { "Behavior0": ["78"], "Behavior1": ["78"], "Behavior2": ["78"], "Behavior3": ["78"], "Behavior4": ["78"], "Behavior5": ["78"], } |
| `device_data_arrays` | `dict or None` | Mapping of friendly names to ``(device, register)`` paths. CameraStart Default: { "Behavior0_StartCameras": {"l0_selector": "Behavior0", "l1_selector": "78"}, "Behavior1_StartCameras": {"l0_selector": "Behavior1", "l1_selector": "78"}, "Behavior2_StartCameras": {"l0_selector": "Behavior2", "l1_selector": "78"}, "Behavior3_StartCameras": {"l0_selector": "Behavior3", "l1_selector": "78"}, "Behavior4_StartCameras": {"l0_selector": "Behavior4", "l1_selector": "78"}, "Behavior5_StartCameras": {"l0_selector": "Behavior5", "l1_selector": "78"}, } |
| `rekey` | `bool` | If True, rekey HARP stems to register keys (e.g. convert ``Behavior0_78_1904-01-01T01-00-00.bin`` into ``78``) so the device tree becomes ``device -> register -> dataframe``. Set to False to keep original keys. |
| `verbose` | `bool` | If True, print verbose messages during initialisation. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from device_data_arrays. Accessed via device.data_arrays['{friendly_name}'] Default CameraStart data_arrays: { "Behavior0_StartCameras": xr.DataArray for Behavior0 register 78, "Behavior1_StartCameras": xr.DataArray for Behavior1 register 78, "Behavior2_StartCameras": xr.DataArray for Behavior2 register 78, "Behavior3_StartCameras": xr.DataArray for Behavior3 register 78, "Behavior4_StartCameras": xr.DataArray for Behavior4 register 78, "Behavior5_StartCameras": xr.DataArray for Behavior5 register 78, } |
</details>

---

#### `Device(...)`
> **class**

Minimal device wrapper for HARP-style binary data. This class only adds the device-specific behavior that MonoSource should not own directly: - use ``read_harp_bin`` for ``.bin`` files - select device folders by prefix - collapse duplicated device folder levels - rekey HARP stems to register keys - optionally warn about missing expected devices or registers - optionally build named device_data_arrays

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `dfs_dict` | `dict or None` | Pre-loaded nested data tree. If provided, directory loading is skipped. |
| `experiment_directory_path` | `str or Path or None` | Root path passed to MonoSource when loading from disk. |
| `harp_device_yaml_path` | `str or Path` | YAML definition passed through to ``read_harp_bin``. |
| `device_type` | `str` | Prefix used to select relevant top-level device folders. |
| `device_list` | `list[str] or None` | Expected device names. Used only for warning messages. |
| `device_IDs` | `dict or None` | Placeholder metadata for expected device IDs. |
| `device_registers` | `dict or None` | Expected registers per device. Also used as an optional filter. |
| `device_data_arrays` | `dict or None` | Mapping of friendly names to ``(device, register)`` paths. |
| `rekey` | `bool` | If True, rekey HARP stems to register keys (e.g. convert ``Behavior0_8_timestamp`` into ``8``) so the device tree becomes ``device -> register -> dataframe``. Set to False to keep original keys. |
| `verbose` | `bool` | If True, print warnings for missing expected content and info about loading steps. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from device_data_arrays. Accessed via device.data_arrays['{friendly_name}'] |
</details>

---

#### `MultiDevice(...)`
> **class**

Device-oriented MultiSource wrapper. Like Device, this class focuses on obtaining the right dfs_dict shape. Unlike Device, it then delegates to MultiSource to construct virtual-map aligned outputs (data_arrays, lookup_arrays, lookup_virtual_coords).

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_directory_path` | `str or Path` | Root directory to walk for HARP data. |
| `harp_device_yaml_path` | `str or Path` | Path to HARP device YAML for parsing device/register metadata. |
| `device_type` | `str` | Device type to filter for in the HARP device YAML. |
| `device_list` | `list of str` | List of device names to include. If None, include all devices of the specified type. |
| `device_IDs` | `dict` | Optional mapping of device names to device IDs for filtering. |
| `device_registers` | `dict` | Optional mapping of device names to lists of register IDs to include. |
| `virtual_maps` | `dict or None` | Optional mapping of virtual map names to virtual maps for lookup construction. |
| `data_keys` | `list or None` | Optional list of keys in virtual_maps to use for data array construction. If None, use all keys. |
| `global_coord_name` | `str` | Name of the global coordinate dimension in the lookup array. |
| `virtual_coord_names` | `list of str` | Names of the virtual coordinate dimensions in the lookup array. |
| `dict_of` | `str` | Format for virtual_map input: 'dicts' for dict-of-dicts, 'tuples' for dict-of-tuples. |
| `data_array_names` | `dict or None` | Optional custom names for data arrays keyed by data_key. e.g. {'Activations': 'my_activations_data'} |
| `data_array_attrs` | `dict or None` | Optional dict of attributes to set on constructed data arrays. |
| `rekey` | `bool` | If True, rekey the dfs_dict to use device names instead of IDs for easier navigation. If False, keep original keys. Default is True. |
| `lookup_array_names` | `dict or None` | Optional custom names for lookup arrays keyed by data_key. e.g. {'Activations': 'my_activations_lookup'} |
| `lookup_array_attrs` | `dict or None` | Optional dict of attributes to set on constructed lookup arrays. |
| `test_values` | `bool` | If True, populate arrays with human-readable debug strings instead of NaNs. |
| `fill_value` | `any` | Value to use for filling missing entries in the lookup array. Default is None. |
| `verbose` | `bool` | If True, print detailed information during loading and construction. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `dfs_dict` | `dict` | Nested dictionary of DataFrames from collect_harp_dfs. |
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from virtual maps. |
| `lookup_arrays` | `dict[str, xr.DataArray]` | Named lookup DataArrays built from virtual maps. |
| `lookup_virtual_coords` | `dict[str, list]` | Virtual coordinate names for each lookup array. |
</details>

---

#### `Nosepoke(...)`
> **class**

Preset MultiDevice for common nosepoke peripheral data configuration. Pre-configures a 6-board, 18-nosepoke layout with 4 data types (Activations, LEDs, Valves, Rewards). Each board has 3 nosepokes. All defaults can be overridden.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_directory_path` | `str or Path or None` | Root path for loading from disk. |
| `harp_device_yaml_path` | `str or Path` | Path to HARP device YAML schema. |
| `device_list` | `list[str]` | Device names. Default: ['Behavior0', ..., 'Behavior5']. |
| `device_registers` | `dict` | Registers per device. Default: {'Behavior0': ['32', '34'], ...}. |
| `virtual_maps` | `dict` | Virtual maps for all data keys. Default maps NP_0..NP_17 across 6 Behavior boards with 3 ports each per data key. |
| `data_keys` | `list[str]` | Data keys to construct. Default: ['Activations', 'LEDs', 'Valves', 'Rewards']. |
| `global_coord_name` | `str` | Name of the global coordinate dimension. Default: 'peripherals'. |
| `virtual_coord_names` | `list[str]` | Virtual coordinate names. Default: ['device', 'register', 'localID']. |
| `rekey` | `bool` | If True, rekey HARP file stems to register addresses. Default: True. |
| `verbose` | `bool` | If True, print progress during loading and construction. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `dfs_dict` | `dict` | Nested dictionary of DataFrames from collect_harp_dfs. |
| `data_arrays` | `dict[str, xr.DataArray]` | One DataArray per data key (Activations, LEDs, Valves, Rewards), each with dims [Time × peripherals] and virtual coords attached. Queryable via da.ulookup.select(device='Behavior0'). |
| `lookup_arrays` | `dict[str, xr.DataArray]` | One lookup array per data key. |
| `lookup_virtual_coords` | `dict[str, dict]` | Unique virtual coordinate values for each data key. |
</details>

---

#### `SoundCard(...)`
> **class**

Preset device wrapper for SoundCard data. This preset provides default SoundCard devices, registers, and named device_data_arrays. It adds no extra helper methods beyond the base class.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_directory_path` | `str or Path or None` | Root path passed to MonoSource when loading from disk. |
| `harp_device_yaml_path` | `str or Path` | YAML definition passed through to ``read_harp_bin``. |
| `device_list` | `list[str] or None` | Expected device names. Used only for warning messages. SoundCard Default: ["SoundCard"] |
| `device_IDs` | `dict or None` | Placeholder metadata for expected device IDs. SoundCard Default: {"SoundCard": "SoundCard0"} |
| `device_registers` | `dict or None` | Expected registers per device. Also used as an optional filter. SoundCard Default: {"SoundCard": ["8", "32", "33", "35"]} |
| `device_data_arrays` | `dict or None` | Mapping of friendly names to ``(device, register)`` paths. SoundCard Default: { "PlaySoundFreq": {"l0_selector": "SoundCard", "l1_selector": "32"}, "StopLog": {"l0_selector": "SoundCard", "l1_selector": "33"}, "AttenuationRight": {"l0_selector": "SoundCard", "l1_selector": "35"}, } |
| `rekey` | `bool` | If True, rekey HARP stems to register keys (e.g. convert ``SoundCard_8_1904-01-01T01-00-00.bin`` into ``32``) so the device tree becomes ``device -> register -> dataframe``. Set to False to keep original keys. |
| `verbose` | `bool` | If True, print verbose messages during initialisation. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_arrays` | `dict[str, xr.DataArray]` | Named DataArrays built from device_data_arrays. Accessed via device.data_arrays['{friendly_name}'] Default SoundCard data_arrays: { "PlaySoundFreq": xr.DataArray for SoundCard register 32, "StopLog": xr.DataArray for SoundCard register 33, "AttenuationRight": xr.DataArray for SoundCard register 35, } |
</details>

---

### MultiSource — `data_conduit.datasources.multisource`
```python
import data_conduit.datasources.multisource
```

Orchestration layer for combining data from multiple devices or files into unified xarray structures using virtual coordinate maps. `MultiSource` is the base class; see `data_conduit.datasources.presets.harp` for HARP-specific multi-device presets (`MultiDevice`, `Nosepoke`).

---

#### `MultiSource(...)`
> **class**

Base class for combined / virtual-array outputs in data-conduit. MultiSource mirrors the top-level flow of MonoSource. It either accepts a pre-built dfs_dict or builds one via collect_dfs, then constructs combined outputs by aligning multiple inputs through virtual maps.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `dfs_dict` | `dict or None` | Optional pre-built nested dictionary of DataFrames/DataArrays. If provided, collect_dfs is not called and this dict is used directly. |
| `experiment_directory_path` | `str or Path` | Root directory to walk when dfs_dict is not provided. |
| `readers` | `dict[str, Callable] or None` | Mapping of file extensions to reader functions. |
| `reader_kwargs` | `dict[str, dict] or None` | Mapping of file extensions to kwargs for each reader. |
| `keep_empty` | `bool` | If True, preserve empty subdirectories as empty dicts. |
| `flatten` | `bool` | If True, flatten the output dict from collect_dfs. |
| `separator` | `str` | Separator used when flatten=True. |
| `virtual_maps` | `dict or None` | Mapping of data_keys to virtual maps. |
| `data_keys` | `list[str] or None` | Data keys to process. If None, uses virtual_maps.keys(). |
| `global_coord_name` | `str` | Name of the global coordinate dimension. |
| `virtual_coord_names` | `list[str] or None` | Names of virtual coordinate dimensions. If None, inferred from the first entry of the first virtual map when possible. |
| `dict_of` | `str` | Format of virtual_map entries: 'dicts' or 'tuples'. |
| `data_array_names` | `dict or None` | Optional custom names for data arrays keyed by data_key. |
| `data_array_attrs` | `dict or None` | Optional attrs applied to data arrays. |
| `lookup_array_names` | `dict or None` | Optional custom names for lookup arrays keyed by data_key. |
| `lookup_array_attrs` | `dict or None` | Optional attrs applied to lookup arrays. |
| `test_values` | `bool` | If True, populate data arrays with readable test values. |
| `fill_value` | `any` | Fill value used for missing entries in populated arrays. |
| `verbose` | `bool` | If True, print progress during loading and construction. |
</details>

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `dfs_dict` | `dict` | Nested dictionary of DataFrames/DataArrays. |
| `virtual_maps` | `dict` | Virtual map configuration keyed by data_key. |
| `data_keys` | `list[str]` | Data keys to construct. |
| `data_arrays` | `dict[str, xr.DataArray]` | Constructed virtual data arrays. |
| `lookup_arrays` | `dict[str, xr.DataArray]` | Constructed lookup arrays. |
| `lookup_virtual_coords` | `dict[str, dict]` | Unique virtual coordinate values for each data key. |
</details>

---

### Virtual Arrays — `data_conduit.virtualarrays`
```python
import data_conduit.virtualarrays
```

Construct and query n-dimensional xarray DataArrays where *virtual coordinates* (e.g. device, register, channel) map onto a single *global coordinate* axis. `ulookup` provides flexible selection by any combination of virtual coordinates, and the `.ulookup()` xarray accessor makes this available directly on DataArrays.

---

#### `LookupAccessorConstructor(data_array: xarray.core.dataarray.DataArray)`
> **class**

Base class for constructing custom xarray DataArray accessors for lookup operations with auto-construction of lookup arrays.

---

#### `construct_data_array(...)`
> **func**

Construct a base DataArray for a given set of global coordinates for all timepoints and their associated virtual coordinates. virtual_map formats: dict_of="dicts":  {global: {vcoord_name: vcoord_value, ...}, ...}

---

#### `construct_lookup_array(...)`
> **func**

Construct a lookup DataArray for a given set of global coordiantes and their associated virtual coordinates. virtual_map formats: dict_of="dicts":  {global: {vcoord_name: vcoord_value, ...}, ...}

---

#### `ulookup(...)`
> **func**

Selects global coordinates from a 1D lookup DataArray based on specified criteria across virtual coordinate dimensions. Supports both positional- and label-based indexing and combinations thereof. Supports: Single values: device='Behavior1' Lists: device=['Behavior0', 'Behavior2'] Slices: device=slice('Behavior1', 'Behavior5')

---

#### `update_data_array(...)`
> **func**

Update DataArray with values from dfs_dict based on mapping provided in virtual_map. Format of dfs_dict and virtual_map should match, i.e. if dfs_dict is indexable as

---

### Segmentation — `data_conduit.segment`
```python
import data_conduit.segment
```

`get_segment` filters a DataFrame to rows where a lookup column falls within a given range, returning the corresponding values from a second column as a NumPy array. Typically used to extract global-clock indices covering a trial window, which are then passed to `.sel()` or `.loc[]` to slice a DataArray aligned to the same clock.

---

#### `get_segment(...)`
> **func**

Extract values from one column where another column falls within a range. Filters ``data`` to rows where ``lookup_column`` is between ``start`` and ``end`` (inclusive), then returns the corresponding values from ``return_column`` as a NumPy array. A typical use case is extracting global-clock index values that fall within a trial window, which can then be passed to ``.sel()`` or ``.loc[]`` to slice a DataArray or DataFrame aligned to the same global clock.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | DataFrame containing both ``lookup_column`` and ``return_column``. |
| `start` | `float` | Start of the range (inclusive) applied to ``lookup_column``. |
| `end` | `float` | End of the range (inclusive) applied to ``lookup_column``. |
| `lookup_column` | `str` | Name of the column to filter on (e.g. a time or global-clock column). |
| `return_column` | `str` | Name of the column whose values are returned for the matching rows. |
</details>

**Returns:** `np.ndarray` — Values from ``return_column`` for all rows where ``lookup_column`` is within ``[start, end]``. Pass this directly to ``.sel()``, ``.loc[]``, or another index lookup to extract the corresponding data from a separate stream aligned to the same column.

---

---
## 2. Synchronisation & Alignment


### TTL Sync — `data_conduit.sync`
```python
import data_conduit.sync
```

End-to-end TTL synchronisation pipeline: extract pulse segments from raw waveforms, align pulse tables across clocks, fit a linear timebase model, and apply the conversion. `TTLSyncModel` encapsulates the fitted slope/intercept/R²; `get_npx_to_bonsai_time_conversion` is a semantic shortcut for the common Neuropixels ↔ Bonsai alignment. Includes visualisation helpers for inspecting pulse alignment and conversion error.

---

#### `TTLSyncModel(slope: float, intercept: float, r2: float) -> None`
> **class**

Simple linear TTL synchronisation model.

<details><summary>Attributes</summary>

| Attribute | Type | Description |
|-----------|------|-------------|
| `slope` | `float` | Scale factor mapping target clock units to reference units. |
| `intercept` | `float` | Offset term in reference units. |
| `r2` | `float` | Coefficient of determination from fit. |
</details>

---

#### `align_pulse_tables(...)`
> **func**

Align reference and target pulse tables for one-to-one model fitting. The function truncates both tables to equal length (`min(len(ref), len(target))`) so each row index corresponds to the same pulse ordinal in both clocks.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_df` | `pd.DataFrame` | Pulse table in reference clock units. Must contain `Start` and `End`. |
| `target_df` | `pd.DataFrame` | Pulse table in target clock units. Must contain `Start` and `End`. |
| `normalise_start` | `bool` | If True, subtract each table's first Start value from Start/End. |
</details>

**Returns:** `tuple[pd.DataFrame, pd.DataFrame]` — Aligned copies of `(reference_df, target_df)` with matching row counts.

---

#### `build_pulse_table(...)`
> **func**

Pair rise and fall timestamps into a pulse table. For each rise, the immediately following fall is found via searchsorted. This correctly handles recordings that start mid-pulse (where a fall arrives before the first rise) — those orphaned falls are skipped.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `rise_times` | `array-like` | Timestamps of rising edges (pulse starts). |
| `fall_times` | `array-like` | Timestamps of falling edges (pulse ends). |
| `min_duration` | `float` | Pulses shorter than this are dropped. Default 0.0 (keep all valid pairs). |
</details>

**Returns:** `pd.DataFrame` — Columns ``Start``, ``End``, ``Duration`` with a 1-based ``Pulse`` index.

---

#### `convert_timebase(...)`
> **func**

Apply a linear timebase transform (`slope * value + intercept`).

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | `np.ndarray \| pd.Series \| list[float]` | Values to be transformed. |
| `slope` | `float` | Linear slope. |
| `intercept` | `float` | Linear intercept. |
</details>

**Returns:** `np.ndarray` — Transformed values.

---

#### `extract_ttl_segments(...)`
> **func**

Extract run-length TTL pulse segments from a sampled waveform.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `times` | `array-like` | Monotonic timestamps corresponding to `values`. |
| `values` | `array-like` | Signal values used to infer active/inactive TTL state. |
| `threshold` | `float` | Threshold used to binarize the signal into state 0/1. |
| `active_high` | `bool` | If True, active state is `value >= threshold`. If False, active state is `value < threshold`. |
| `min_duration` | `float` | Minimum segment duration. Segments shorter than this are dropped. |
| `drop_inactive` | `bool` | If True, keep only active segments (`State == 1`). |
| `align_to_zero` | `bool` | If True, subtract the first kept segment start from Start/End, so the first segment starts at 0. |
</details>

**Returns:** `tuple[pd.DataFrame, float]` — (`segments`, `offset`) where `segments` has columns ['Start', 'End', 'Duration', 'State'] and `offset` is the original start value used for alignment.

---

#### `fit_linear_timebase(...)`
> **func**

Fit a linear model that maps target clock values into reference clock values.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_df` | `pd.DataFrame` | Reference-clock pulse table. |
| `target_df` | `pd.DataFrame` | Target-clock pulse table. |
| `use` | `str` | Edge used for fitting: `'start'` or `'end'` (case-insensitive). |
</details>

**Returns:** `dict[str, float]` — Dictionary with keys `slope`, `intercept`, and `r2`.

---

#### `get_npx_to_bonsai_time_conversion(...)`
> **func**

Convert NPX pulse times into Bonsai time using TTL pulse alignment. This is a semantic convenience wrapper over `get_ttl_timebase_conversion` with labels prefilled for NPX/Bonsai workflows.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_pulses` | `pd.DataFrame` | Bonsai-clock pulse table. |
| `target_pulses` | `pd.DataFrame` | NPX-clock pulse table to be converted into Bonsai time. |
| `use` | `str` | Edge column used for fitting ('start' or 'end'). |
| `normalise_start` | `bool` | If True, align both pulse tables to zero at first pulse before fitting. |
| `visualisation` | `bool` | If True, render Workflow_v2-style stacked pulse visualisation. |
| `error_visualisation_tools` | `bool` | If True, render per-pulse error scatter/hist diagnostics. |
| `return_details` | `bool` | If True, additionally return fitted `TTLSyncModel` and fit stats. |
</details>

---

#### `get_ttl_timebase_conversion(...)`
> **func**

Convert a target pulse table into a reference clock domain. This helper performs: 1) aligning pulse tables, 2) fitting a linear target->reference model, 3) returning converted target pulse timestamps + conversion ratio.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_pulses` | `pd.DataFrame` | Reference-clock pulse table. |
| `target_pulses` | `pd.DataFrame` | Target-clock pulse table to be converted into reference time. |
| `use` | `str` | Edge column used for fitting ('start' or 'end'). |
| `normalise_start` | `bool` | If True, align both pulse tables to zero at first pulse before fitting. |
| `visualisation` | `bool` | If True, render Workflow_v2-style stacked pulse visualisation. |
| `error_visualisation_tools` | `bool` | If True, render per-pulse error scatter/hist diagnostics. |
| `reference_label` | `str` | Display label for reference pulses in stacked plots. |
| `target_label` | `str` | Display label for target pulses in stacked plots. |
| `return_details` | `bool` | If True, additionally return fitted `TTLSyncModel` and fit stats. |
</details>

**Returns:** `tuple` — Default: (`converted_target_df`, `conversion_ratio`) If `return_details=True`: (`converted_target_df`, `conversion_ratio`, `model`, `fit_stats`)

---

#### `plot_conversion_error_tools(reference_pulses: 'pd.DataFrame', converted_target_pulses: 'pd.DataFrame')`
> **func**

Per-pulse conversion error diagnostics: scatter plots and histograms. Computes the residual between converted target pulse times and reference pulse times for Start, End, and Duration. Renders a 3×2 grid of scatter plots (error vs pulse index) and histograms (error distribution) in milliseconds.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_pulses` | `pd.DataFrame` | Reference-clock pulse table. Must contain ``Start``, ``End``, and ``Duration``. |
| `converted_target_pulses` | `pd.DataFrame` | Target pulse table after conversion to the reference clock domain. Must contain ``Start``, ``End``, and ``Duration``. |
</details>

**Returns:** `tuple[matplotlib.figure.Figure, numpy.ndarray]` — ``(fig, axs)`` where ``axs`` is a 3×2 array of axes.

---

#### `plot_stacked_pulses(...)`
> **func**

Stacked broken-bar visualisation of reference and converted target pulses. Renders reference (blue) and target (red) pulses as horizontal broken bars on dual x-axes, wrapping across multiple rows. Both streams must be in the same time domain (i.e. pass converted target pulses, not raw target pulses). The target axis is a twin of the reference axis. Its background patch is made transparent so that only the reference axis background is visible.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_pulses` | `pd.DataFrame` | Reference-clock pulse table. Must contain ``Start`` and either ``End`` or ``Duration``. |
| `target_pulses` | `pd.DataFrame` | Converted target pulse table (same time domain as reference). Must contain ``Start`` and either ``End`` or ``Duration``. |
| `pulses_per_row` | `int or None` | Number of pulses to show per row. If ``None``, determined automatically from ``autofit_pulses_per_row``. |
| `autofit_pulses_per_row` | `bool` | If ``True`` and ``pulses_per_row`` is ``None``, selects a round number of pulses per row from a fixed set of candidates. Default ``True``. |
| `reference_label` | `str` | Display label for the reference stream. Default ``'Reference'``. |
| `target_label` | `str` | Display label for the target stream. Default ``'Target'``. |
</details>

**Returns:** `tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]] or tuple[None, None]` — ``(fig, axes)`` — or ``(None, None)`` if there are no pulses to plot.

---

#### `plot_timebase_fit(...)`
> **func**

Plot the linear target-to-reference timebase fit. Scatter-plots matched pulse edge times (target on x, reference on y) and overlays the fitted linear model. Provides a visual check of linearity and goodness-of-fit.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_pulses` | `pd.DataFrame` | Reference-clock pulse table. Must contain ``Start`` and ``End``. |
| `target_pulses` | `pd.DataFrame` | Target-clock pulse table. Must contain ``Start`` and ``End``. |
| `use` | `str` | Edge used for fitting and plotting: ``'start'`` or ``'end'``. Default ``'start'``. |
| `model` | `TTLSyncModel or None` | Pre-fitted model. If ``None``, a model is fitted from the supplied pulse tables. |
| `ax` | `matplotlib.axes.Axes or None` | Axes to draw on. If ``None``, a new figure is created. |
</details>

**Returns:** `matplotlib.axes.Axes` — The axes containing the plot.

---

#### `plot_ttl_pulse_trains(...)`
> **func**

Plot aligned TTL pulse trains with target stream vertically offset. Both streams are normalised to start at zero before plotting. Useful as a quick visual sanity check that the two pulse trains are structurally similar before fitting a conversion model.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_pulses` | `pd.DataFrame` | Reference-clock pulse table (e.g. Bonsai/HARP). Must contain ``Start`` and ``End`` columns. |
| `target_pulses` | `pd.DataFrame` | Target-clock pulse table (e.g. Neuropixels). Must contain ``Start`` and ``End`` columns. |
| `max_pulses` | `int` | Maximum number of pulses to render per stream. Default 80. |
| `target_vertical_offset` | `float` | Vertical offset applied to the target stream so the two trains do not overlap. Default 1.25. |
| `reference_color` | `str` | Matplotlib colour string for reference pulses. Default ``'tab:blue'``. |
| `target_color` | `str` | Matplotlib colour string for target pulses. Default ``'tab:orange'``. |
| `alpha` | `float` | Opacity of pulse outlines. Default 0.35. |
| `ax` | `matplotlib.axes.Axes or None` | Axes to draw on. If ``None``, a new figure is created. |
</details>

**Returns:** `matplotlib.axes.Axes` — The axes containing the plot.

---

### Global Times — `data_conduit.globaltimes`
```python
import data_conduit.globaltimes
```

Create a canonical, evenly-spaced global time vector and map heterogeneous stream timestamps onto it. `index_map_util` supports nearest, before, after, and exact matching strategies.

---

#### `create_global_clock(...)`
> **func**

Create a global clock as a 1D array of timestamps from start_time to end_time with specified intervals.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_time` | `float or int or None` | Start time of the global clock. If None, defaults to 0.0. |
| `end_time` | `float or int` | End time of the global clock. |
| `timestep_interval` | `float or int` | Time interval between consecutive timestamps in the global clock. Must be non-zero. |
| `include_end_time` | `bool, optional` | If True, include end_time in the global clock if it falls on a valid timestep. Default is True. |
</details>

**Returns:** `ndarray` — 1D array of timestamps representing the global clock, starting from start_time up to end_time with intervals of timestep_interval. If include_end_time is True and end_time does not fall on a valid timestep, end_time will be included as the last timestamp in the array.

---

#### `index_map_util(...)`
> **func**

Map stream timestamps to global clock timestamps.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `global_clock_times` | `array-like` | 1D array of global clock timestamps. |
| `stream_times` | `array-like or pd.DataFrame or xr.DataArray` | 1D array or single-column DataFrame or 1D DataArray of stream timestamps. |
| `match_type` | `{'nearest', 'before', 'after', 'exact'}, optional` | Matching rule for mapping stream times to global times. Default is 'nearest'. |
</details>

**Returns:** `dict` — { 'index_position_array': DataFrame [N x 2] -> columns ['global_index', 'stream_index'], 'matched_global_times': DataFrame [N x 2] -> columns ['global_time', 'stream_time'], 'global_stream_time_delta': Series [N] -> (global_time - stream_time), 'full': DataFrame [N x 5] -> columns ['global_index', 'stream_index', 'global_time', 'stream_time', 'delta'] } - 'index_position_array': DataFrame where each row contains the global index and the corresponding stream index (-1 if no match). - 'matched_global_times': DataFrame where each row contains the global time and the matched stream time (NaN if no match). - 'global_stream_time_delta': Series containing the difference between global time and matched stream time (NaN if no match). - 'full': DataFrame where each row contains global_index, stream_index, global_time, stream_time, and delta.

---

---
## 3. Internals & Utilities


### Timestamps — `data_conduit.timestamps`
```python
import data_conduit.timestamps
```

Collect timestamp vectors from individual DataFrames, flat dictionaries, or arbitrarily nested data structures. Supports flexible lookup by column name, index, or auto-detection.

---

#### `collect_timestamps(...)`
> **func**

Collect and process timestamps from a given DataFrame.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame \| xr.DataArray \| None` | The DataFrame or DataArray containing the timestamps. |
| `timestamp_name` | `str \| None` | The name of the column containing the timestamps. If None, defaults to 'Time'. Note: case-insensitive. |
| `time_location` | `str \| None` | The location of the timestamps, either 'columns' or 'index'. If None, will attempt index first, then columns. |
| `verbose` | `bool` | If True, prints warnings and information during processing. |
</details>

**Returns:** `pd.Series | None` — A Series of processed timestamps, or None if the DataFrame is None.

---

#### `collect_timestamps_dict(...)`
> **func**

Extract timestamps from each DataFrame in a dictionary. Parameters: dfs (dict): A dictionary where keys are identifiers and values are pandas DataFrames. timestamp_name (str, optional): The name of the column containing the timestamps. If None, defaults to 'Time'. Note: case-insensitive. verbose (bool, optional): If True, prints warnings and information during processing. Default is True. return_type (str, optional): If 'list', returns a sorted list of all timestamps. If 'dict', returns a dictionary of timestamps per DataFrame. Default is 'list'. If 'both', returns two outputs: a sorted list of all timestamps and a dictionary of timestamps per DataFrame. Returns: list of all timestamps from each DataFrame in the dictionary, sorted in ascending order. dict where keys are the same as df_dict and values are the timestamps from each DataFrame.

---

#### `collect_timestamps_nested(...)`
> **func**

Collect timestamps from an arbitrarily nested dictionary of DataFrames/DataArrays. This function first flattens the nested dictionary structure, then applies the base timestamp collection logic to the flattened dictionary.

---

### HarpTools — `data_conduit.harptools`
```python
import data_conduit.harptools
```

Extends the `harp-python` library with device reader construction from YAML schemas, register address lookup, and a `read_harp_bin` reader that plugs into the IO registry.

---

#### `collect_harp_dfs(...)`
> **func**

HARP-flavoured wrapper around collect_dfs. Walks the experiment directory for .bin files under device_type folders, reads them using read_harp_bin, then applies HARP-specific cleanup: 1. Collapse Bonsai double-folder structure (Device/Device/files → Device/files) 2. Rekey ugly file stems to register address strings (if rekey=True) 3. Optionally filter to only expected registers 4. Optionally warn about missing expected devices/registers

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_path` | `str or Path` | Root experiment directory. |
| `harp_device_yaml_path` | `str or Path` | Path to the HARP device YAML schema. |
| `device_type` | `str` | Folder prefix to select (e.g. 'Behavior', 'SoundCard'). |
| `device_registers` | `dict or None` | ``{device_name: [register_addresses]}`` to filter and validate. If None, all registers found are kept. |
| `device_list` | `list[str] or None` | Expected device names. Used only for warning messages. |
| `rekey` | `bool` | If True (default), rekey file stems to register address strings. If False, keep the original file stems as dict keys. |
| `keep_empty` | `bool` | If True, preserve empty subdirectories as empty dicts. |
| `verbose` | `bool` | If True, print warnings for missing devices/registers. |
</details>

**Returns:** `dict` — Clean nested dictionary: ``{device_name: {register_address: pd.DataFrame}}``. Register address keys are strings (e.g. '32', '34') if rekey=True, or raw file stems if rekey=False.

---

#### `collect_registers(...)`
> **func**

Find register name(s) by address(es) in the schema file. Efficiently loads the YAML file once for multiple lookups. Parameters: harp_device_yaml_path: yaml file path register_addresses: single int/string OR list of ints/strings verbose: If True, prints additional information. Returns: Single string (if input was scalar) OR List of strings (if input was list)

---

#### `construct_device_reader(harp_device_yaml_path: str | pathlib.Path)`
> **func**

Create a harp reader for a specific device using the provided YAML configuration file.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `harp_device_yaml_path` | `str \| Path` | Path to the YAML configuration file for the device. |
</details>

**Returns:** `harp.Reader` — A harp reader object for the specified device.

---

#### `read_harp_bin(...)`
> **func**

Read a HARP .bin file into a pandas DataFrame. Parses the register address from the filename, resolves it to a register name using collect_registers, and reads using a cached device reader from construct_device_reader.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str or Path` | Path to the .bin file. Expected filename format: {DeviceName}_{RegisterAddress}_{Timestamp}.bin |
| `harp_device_yaml_path` | `str or Path` | Path to the YAML configuration file for the device. |
</details>

**Returns:** `pd.DataFrame` — DataFrame with Time as the index.

---

### Utils — `data_conduit.utils`
```python
import data_conduit.utils
```

Helper callables for level selectors (`starts_with`, `ends_with`, `contains`) and internal directory-traversal / nested-dict utilities.

---

#### `_apply_level_selectors(d: dict, selectors: dict[int, list], current_depth: int = 0) -> dict`
> **func**

Return a filtered copy of `d`, keeping only keys listed in `selectors[depth]` at each depth.

---

#### `_attempt_read(item, readers, reader_kwargs, verbose)`
> **func**

Attempt to read file with matching reader. Returns DataFrame, Path, or None if no reader matches or read fails.

---

#### `_flatten_nested_dict(...)`
> **func**

Recursively flattens arbitrarily nested dictionaries of DataFrames or DataArrays. Keys are concatenated using the specified separator to create unique identifiers for each DataFrame/DataArray in the flattened structure.

---

#### `_get_nested_dict_depth(d: dict) -> int`
> **func**

Return the maximum nesting depth of a dictionary (1 = flat dict with no nested dicts).

---

#### `_is_readable(item, valid_extensions)`
> **func**

Check if file has matching reader. If no readers defined, read everything.

---

#### `_matches_selector(key: str, selector: list | collections.abc.Callable | str | None) -> bool`
> **func**

Check if a key matches a selector. Callable is used to allow for flexible matching logic (e.g., regex, custom functions) beyond simple list or string matching. If selector is None, it matches everything. Base helper callables: - starts_with(prefix): Return true if the key starts with the given prefix. - ends_with(suffix): Return true if the key ends with the given suffix. - contains(substring): Return true if the key contains the given substring. Parameters key : str The key to check (folder name or file stem). selector : list, callable, str, or None - None: matches everything (wildcard) - list: key must be in the list - callable: must return True for the key - str: exact match Returns bool: True if the key matches the selector, False otherwise.

---

#### `_parse_selectors(kwargs: dict) -> dict[int, any]`
> **func**

Validate and parse level selector kwargs into a dictionary mapping depth levels to their corresponding selector values.

<details><summary>Parameters</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| `kwargs` | `dict` | Keyword arguments from collect_dfs. Must follow the pattern 'level_{n}_selector' where n is a non-negative integer. |
</details>

**Returns:** `dict[int, any]` — Mapping of {depth_level: selector_value} for each provided level selector. E.g. {'l0_selector': ['folder1', 'folder2'], 'l1_selector': lambda x: x.startswith('data_')} -> {0: ['folder1', 'folder2'], 1: lambda x: x.startswith('data_')}

---

#### `_passes_selector(name, depth, level_selectors)`
> **func**

Check if name passes selector at given level. If no selector for that level, pass everything.

---

#### `_walk_dirtree(...)`
> **func**

*No description available.*

---

#### `contains(substring: str) -> collections.abc.Callable[[str], bool]`
> **func**

Return a function that checks if a string contains the given substring.

---

#### `ends_with(suffix: str) -> collections.abc.Callable[[str], bool]`
> **func**

Return a function that checks if a string ends with the given suffix.

---

#### `starts_with(prefix: str) -> collections.abc.Callable[[str], bool]`
> **func**

Return a function that checks if a string starts with the given prefix.

---

### Validators — `data_conduit.validators`
```python
import data_conduit.validators
```

Shared validation logic used by `MonoSource` and `MultiSource` to resolve whether to build a new `dfs_dict` from a directory or reuse a pre-built one.

---

#### `validate_dfs_dict_input(dfs_dict: dict | None = None, base_path=None) -> bool`
> **func**

Validate the two allowed ways of obtaining a dfs_dict for collect_dfs. Returns True if collect_dfs should be called. Returns False if an existing dfs_dict should be reused.

---
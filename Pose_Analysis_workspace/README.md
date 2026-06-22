# Pose Analysis Workspace

Extract **everything** from a set of Bonsai/HARP sessions — including the
DeepLabCut **pose** data in each session's `DLC/` folder — time-align each
session, and stack every data type **across** sessions into one container keyed
by data type.

```
select_sessions ─▶ filter_sessions ─▶ build_sessions ─▶ combine_sessions
 (which sessions)   (narrow by day/     (read + align     (stack each type
                     name; require DLC)   each session)     across sessions)
```

The result is a plain dict:

```python
selected_sessions = {
    "dlc:position":         <xr.DataArray  (Time × keypoints × space)>,
    "dlc:confidence":       <xr.DataArray  (Time × keypoints)>,
    "events":               <pd.DataFrame>,
    "nosepoke:Activations": <xr.DataArray  (Time × peripherals)>,   # if HARP YAMLs present
    "soundcard:PlaySoundFreq": <xr.Dataset>,                        # ...
    ...
}
```

— one entry per data type, each **concatenated across sessions** and carrying a
`label` coord (xarray) / column (DataFrame) that tags every row/sample with its
source session.

---

## Contents

| File | Purpose |
|---|---|
| [`pose_analysis_workflow.ipynb`](pose_analysis_workflow.ipynb) | The end-to-end notebook. Edit the config cell, run top to bottom. |
| [`pose_analysis_utils.py`](pose_analysis_utils.py) | `filter_sessions` — narrow a selection by `{day}/{session}` substring/regex and require subfolders (e.g. `DLC/`). |
| `../environment.yml` | Conda environment (repo root). Installs data-conduit editable + all deps. |
| `../src/data_conduit/datasources/pose/` | The new `DLCPose` reader added to data-conduit itself. |

---

## What was built, and why

### The workflow already existed
`data_conduit.sessiongroups` already implements the *select → align → combine*
pipeline. We did **not** rebuild it. The relevant pieces:

- `select_sessions(root, depth=…, level_names=…)` — walk a directory tree to the
  session folders and label each (records intermediate levels like `day` as
  metadata).
- `build_sessions(selection, catalog, …)` — run the per-session alignment
  pipeline (`load_session` → optional `normalise_to_zero` → optional
  `attach_cross_clock` → optional `build_global_clock`/`build_index_tables`) and
  return an ordered `SessionGroup`.
- `combine_sessions(group, type_name=None)` — stack **every** member across the
  group and return a `Session` whose `.data` is the container shown above. (Pass
  a `type_name` like `"dlc:position"` to combine just one type.)

So the cross-session container you wanted (`{type: concatenated-with-session-key}`)
is precisely `combine_sessions(group, type_name=None).data`.

### The missing piece: DeepLabCut pose
DLC was only *sketched* in the demos (via the TTL cross-clock path). Your case is
simpler and was implemented fresh in
[`src/data_conduit/datasources/pose/dlc.py`](../src/data_conduit/datasources/pose/dlc.py).

**Why no TTL sync is needed.** DeepLabCut writes one row per video frame, indexed
by frame number — it has *no clock of its own*. But the Bonsai `VideoData` CSV
logs one row per frame too, and its `Seconds` column is the HARP timestamp of
each frame. Crucially, that timestamp is on the **same clock as
ExperimentEvents** (verified on real data: both the events log and the camera
frames start at `6343.248 s` in the session checked). So aligning DLC is just a
**positional join**: DLC row *i* takes the camera frame time of `VideoData` row
*i*. After that join, pose is on the shared session clock and flows through the
ordinary pipeline like any other same-clock stream — its catalog spec has
`sync=None` (no TTL).

**Segments.** A session can split its recording into several file segments
(Bonsai rolls the video/CSV partway through; ExperimentEvents splits the same
way). Each DLC output is paired with the `VideoData` file of **matching frame
count**, and the segments are concatenated in time order. The length match is
also a sanity check — a DLC file whose row count matches no `VideoData` segment
is surfaced (warn or error), because it usually means a dropped-frame mismatch or
a stale/wrong DLC file. (On the test session: `…DLC.h5` = 51,407 frames ↔
`VideoData_…T01.csv` = 51,407; `…_2DLC.h5` = 38,620 ↔ `VideoData_…T02.csv` =
38,620; concatenated → 90,027 frames, monotonic across the boundary.)

**Pose shape.** Per session, `DLCPose` exposes two `xr.DataArray`s via
`.data_arrays`, which become the bundle members `dlc:position` and
`dlc:confidence`:

- `position`  — dims `(Time, keypoints, space)`, where `space ∈ {x, y}`
- `confidence` — dims `(Time, keypoints)` (the DLC likelihood)

This split (position vs. likelihood) matches the
[`movement`](https://movement.neuroinformatics.dev/) package's native schema.
`pose_to_movement(position, confidence)` repackages the pair into a
`movement`-style `xr.Dataset` with dims `(time, individuals, keypoints, space)`.

> **One schema note.** data-conduit aligns on a dimension named `Time`; movement
> expects `time`. `pose_to_movement` renames it at the hand-off, so both worlds
> are satisfied — but if you index pose arrays directly inside data-conduit, use
> `Time`.

### Session selection helper
`select_sessions` filters the *session* level by exact name and *intermediate*
levels by `l{n}_selector`. For pose work we wanted a looser, one-shot filter, so
[`pose_analysis_utils.filter_sessions`](pose_analysis_utils.py) post-filters the
selection dict: keep sessions whose flattened `"{day}/{session}"` string matches
a substring/regex, and/or that contain required subfolders (e.g. `DLC/`). The
`require_dirs=("DLC",)` gate matters before a whole-session combine — see
*Gotchas* below.

### Environment & packaging
- `../environment.yml` — a conda env (`data-conduit`) that installs the core
  deps, PyTables (for reading DLC `.h5`), Jupyter, and via pip `harp-python`,
  `itables`, `movement`, and **data-conduit itself in editable mode** (`-e .`).
  Editable install means `import data_conduit` works against this checkout now
  **and** stays valid unchanged once data-conduit is released to PyPI
  (`pip install data-conduit`).
- `../pyproject.toml` — added optional-dependency groups
  (`harp`, `pose`, `movement`, `notebooks`, `all`) so future installs can pull
  only what a modality needs, e.g. `pip install data-conduit[harp,pose]`.

---

## How to process data — step by step

### 0. One-time: create the environment

From the **repo root** (the folder with `environment.yml` and `pyproject.toml`):

```bash
conda env create -f environment.yml
conda activate data-conduit
```

To rebuild later after changing deps: `conda env update -f environment.yml --prune`.

> **HARP YAMLs.** `device.yml` and `soundcard.yml` are gitignored and live at the
> repo root by convention. They are needed **only** for the HARP streams
> (`nosepoke`, `soundcard`, `camera`). Without them those streams skip with a
> warning and you still get `events`, `video`, and DLC pose. Drop the YAMLs at
> the repo root to enable the HARP streams.

### 1. Open the notebook and set the config

Launch Jupyter (`jupyter lab`) from the repo root, open
[`pose_analysis_workflow.ipynb`](pose_analysis_workflow.ipynb), and edit the
**config cell** (section 0.1):

```python
DATA_ROOT = Path("/media/sepi/Elements/PathIntegrationProtocol/BonsaiOutput")
PHASE     = "Shaping"            # Shaping / Training / Test
MOUSE_ID  = "FbR_M01569522"      # mouse folder under <DATA_ROOT>/<PHASE>
DATA_DIR  = DATA_ROOT / PHASE / MOUSE_ID

SESSION_PATTERN = None           # e.g. "Day1/" (exact Day1) or r"^Day(1|2)/" with SESSION_REGEX=True
SESSION_REGEX   = False
REQUIRE_DLC     = True           # drop sessions with no DLC/ folder
```

The data layout assumed is
`<DATA_ROOT>/<PHASE>/<MOUSE_ID>/<day>/<session>/…`, so `DATA_DIR` points at one
phase+mouse and `select_sessions(DATA_DIR, depth=1, level_names=("day",))` finds
the `day/session` folders below it.

### 2. Select sessions

```python
sessions = select_sessions(DATA_DIR, depth=1, level_names=("day",))
sessions = filter_sessions(
    sessions, SESSION_PATTERN, regex=SESSION_REGEX,
    require_dirs=("DLC",) if REQUIRE_DLC else None,
)
```

- `"Day1/"` matches *exactly* Day1 — the trailing `/` excludes `Day10…Day19`.
- Drop the pattern (leave `None`) to take every session; `require_dirs=("DLC",)`
  still keeps only pose-bearing ones.

### 3. Choose what to extract (the catalog)

```python
catalog = default_harp_catalog(device_yaml=DEVICE_YAML, soundcard_yaml=SOUNDCARD_YAML,
                               include_video=True, include_session_settings=True)
catalog.remove("camera")                       # this rig logs VideoData, not Camera0Frames
catalog.add(DataStructureSpec(                  # add DeepLabCut pose
    name="dlc",
    reader=lambda p: DLCPose(experiment_directory_path=p, on_length_mismatch="warn"),
    required=False,
))
```

Toggle anything: `catalog.disable("video")`, `catalog.remove("soundcard")`,
`catalog.add(...)`. The DLC spec adds the members `dlc:position` and
`dlc:confidence`.

### 4. Build + combine across sessions

```python
group = build_sessions(sessions, catalog, sort_by=("day", "label"), normalise=False)
combined = combine_sessions(group, type_name=None)   # whole-session combine
selected_sessions = combined.data                    # the container you want
```

`normalise=False` keeps every stream on its shared original clock (this is the
default and the intended mode; `normalise_to_zero` is a redundancy option, not
the standard path).

### 5. Work with the result

```python
position   = selected_sessions["dlc:position"]    # (Time × keypoints × space), label coord on Time
confidence = selected_sessions["dlc:confidence"]

# per-session frame counts
pd.Series(position["label"].values).value_counts()

# one session's nose track
fl = str(position["label"].values[0])
nose = position.sel(keypoints="nose").where(position["label"] == fl, drop=True)

# tag every sample with the day it came from
label_to_day = {n: e["day"] for n, e in sessions.items()}
position = add_label_column(position, "day",
    lambda da: np.array([label_to_day[l] for l in da["label"].values]), dim="Time")
```

### 6. Hand off to `movement` (optional)

```python
poses_ds = pose_to_movement(position, confidence, fps=60.0)
# dims (time, individuals, keypoints, space); position + confidence data vars; label coord preserved
```

---

## The resulting data structure

`selected_sessions` is `dict[str, xr.DataArray | xr.Dataset | pd.DataFrame]`:

- **Keys** use the `spec:member` convention — `dlc:position`,
  `nosepoke:Activations`, `soundcard:PlaySoundFreq`. Single-table streams are
  bare: `events`, `video`.
- **Each value** is one object holding *every* session's data for that type,
  concatenated along `Time` (xarray) or row-wise (DataFrame).
- **Provenance** is the `label` coord/column: for any row/sample you can read
  which session it came from, e.g. `position["label"].values`.

Because the combined Nosepoke stays an `xr.DataArray`, its virtual-coordinate
query still works: `selected_sessions["nosepoke:Activations"].ulookup.select(device="Behavior0")`.

---

## Gotchas & notes

- **Combine scope (`on="intersection"`).** `combine_sessions(type_name=None)`
  keeps only members present in **every** session. If one selected session lacks
  DLC, `dlc:*` would silently drop from the combined dict. Requiring a `DLC/`
  folder in step 2 (`REQUIRE_DLC=True`) avoids this. Alternatively combine a
  single type across whatever sessions have it:
  `combine_sessions(group, type_name="dlc:position")`.
- **Length-mismatch handling.** The DLC reader pairs files by frame count. A
  mismatch is a warning in the notebook (`on_length_mismatch="warn"`) so one bad
  session doesn't abort the batch; set `"error"` in the spec to make it fatal for
  strict single-session checks.
- **HARP YAMLs / `harp` package.** `default_harp_catalog` and the HARP presets
  need both the YAMLs *and* the `harp-python` package (imports as `harp`,
  installed by the conda env). Pose/events/video need neither.
- **`Time` vs `time`.** Stay on `Time` inside data-conduit; `pose_to_movement`
  renames to `time` only at the movement boundary.
- **`.h5` vs `.csv`.** The reader prefers `.h5` (faster, exact dtypes) and falls
  back to `.csv`; pin with `DLCPose(..., file_format="csv")`.

---

## Verification done

The DLC reader and the full pipeline were tested against real data on
`…/Shaping/FbR_M01569522`:

- DLC read + frame-time join: 90,027 frames, times on the Bonsai clock,
  monotonic across the segment boundary, exact length pairing.
- `load_session` → `build_sessions` → `combine_sessions(type_name=None)` →
  `selected_sessions` with `dlc:position` / `dlc:confidence` / `events`, the
  `label` coord preserved through `add_label_column` and `pose_to_movement`.
- `ruff check` passes on the new module and helper.

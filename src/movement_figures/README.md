# Movement paper figure templates

Five notebooks and two reusable Python modules, arranged in the existing folders.

| Start here | Purpose |
|---|---|
| [Data import demo](data_template/demo.ipynb) | Configure, preview and load one session; inspect coverage and process pose |
| [Annotation demo](timeseries_template/demo.ipynb) | Synthetic examples of regions, event lines, outcomes, legends and relative time |
| [Kinematics](Figures/Kinematics/kinematics.ipynb) | x/y location, linear speed and angular head velocity; session and trial panels |
| [Trajectories](Figures/Trajectories/trajectories.ipynb) | Single/multiple trials, inbound deviation, outbound straightness and paired comparisons |
| [Spatial](Figures/Spatial/spatial.ipynb) | Allocentric and body-relative head direction; whole-session/outbound/inbound occupancy |

The modules are [loading.py](data_template/loading.py) and
[annotations.py](timeseries_template/annotations.py). No existing QC notebook or
core data-conduit module is changed.

## Running on a session

Use a Python 3.12 notebook kernel with this data-conduit checkout and movement
0.17.0 (the version used for local validation). The loader rejects older movement
versions; later versions have not been validated here. The existing project
environment supplies the notebook/plotting dependencies. DLC HDF5 input also
needs PyTables. These notebooks use `data_conduit.refactor_qc` for the catalog,
trial parsing and DLC alignment, matching the workflow tested on the data machine.
The figure templates themselves still need real-session validation there.

Start Jupyter within this checkout and open the data import demo first. Set
`ROOT`, the exact `SESSION` folder name, and the hierarchy beneath the root.
The default hierarchy is `ROOT / mouseID / day / session`; for `ROOT / session`,
use `LEVEL_NAMES = ()`. Optional level selectors disambiguate repeated names.
Selection must resolve to exactly one session before readers run.

`SOURCES = ("trials", "events", "dlc")` loads parsed trials, raw event markers
and pose. The catalog loads VideoData as DLC's clock-alignment dependency.
The figure loader requires trials and DLC; extra supported sources can be
requested explicitly. It does not load raw nosepoke activations by default.
The shared template adapts these source names to refactor_qc's catalog: its
legacy `events` request produces parsed trials, while an additional reader
retains the raw events for plot annotations. Both use the same ExperimentEvents
reader and split-file concatenation. The existing trial reader is unchanged.

Copy the same root, scope, keypoint and processing settings into each figure
notebook's configuration cell. Inspect the printed keypoint names, confidence
quality, trial table and stream coverage. `body`, `tailbase`, `lear` and `rear`
are editable defaults from the earlier QC work; they must match the recording.
Raw pose is retained separately. Confidence 0.9 is an example setting, while
interpolation and smoothing are disabled until configured.

## Measurement definitions

- Data-conduit handles import, event parsing, time alignment and trial slicing.
  Pose processing and movement measurements use movement functions; xarray
  selects/assembles their inputs, and Matplotlib arranges and annotates figures.
- Linear speed uses `compute_speed`. Angular head velocity composes
  `compute_head_direction_vector` and `compute_signed_angle_2d`, then divides
  consecutive signed rotations by the recorded time intervals. This is an
  explicit calculation using movement primitives, not a dedicated movement
  angular-velocity API. It assumes less than half a turn between adjacent
  samples and estimates the preceding interval's average rotation rate.
- Outbound **path straightness**, D/L, is the proposed operational measure for
  the requested complexity panel. Lower means less direct; it does not measure
  every form of complexity. The function is `compute_path_straightness`.
- `compute_path_deviation` measures inbound distance from the infinite line
  through that observed path's endpoints. It is not error relative to a known
  home port. The displayed mean is an arithmetic mean across samples.
- Allocentric head direction is relative to a fixed image/arena axis.
  **Egocentric is provisionally defined as head relative to body.** A target's
  bearing relative to the head would require a target/ROI definition instead.
- Occupancy uses `plot_occupancy` and reports valid sample counts per bin, not
  time spent. Phase panels share bin edges and colour limits.

Positions remain pixels; time remains absolute acquisition seconds. Image
coordinates have +y down, so positive head angles/rotation are clockwise.
Raw `Start Trial` events and derived trial boundaries are distinct. Misses end
with a trial-end marker; a decision poke requires a valid chosen port. Missing
target triggers omit phase regions/metrics and appear in the omission reports.
Long acquisition-clock gaps still need review even if positions contain no NaNs.

## Validation and remaining choices

Local checks use an explicitly synthetic catalog passed through the actual
single-session loader. The figure cells have been executed and their rendered
plots inspected. Focused checks cover absolute time, coverage, bounded processing,
angle wrapping/sign, missing head vectors, annotation timing and incomplete
phase coverage. The standalone annotation demo also runs entirely on synthetic
data. Notebook files retain no execution outputs.

These are runnable templates, not validated experimental results. Set the real
data root/session, confirm the body-relative definition, and review the keypoints
and processing settings before running each notebook from a fresh kernel.

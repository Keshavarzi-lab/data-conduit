# movement_figures

These notebooks read the lab's recordings through the existing data-conduit `DataStructure`, select data before plotting, and call movement for measurements.

Start with `data_template/demo.ipynb`. Set your recording root, hierarchy and directory selectors; preview `.select()`, then call `.load()`. Its outputs are ordinary pandas/xarray streams. No `FigureData`, `SessionData`, `build_config`, or `load_figure_data` interface is required.

## Recording, trial and condition selection

`build_datastructure` uses explicit stream names: `trials` returns the parsed trial table and `events` returns its raw event log. Both share one source read. DLC can load without events; it needs the corresponding video timestamps for alignment.

Use `slice_stream(data, selectors={...}, where=...)` for direct selection. A session or trial table is not required. Selected trial rows can be passed to `slice_stream_for_trial`: each row supplies its recording, bounds and inclusion flags. Original trial numbers are retained independently within each recording. Parse the complete event log before filtering: event-ID ranges preserve tied-event boundaries, but cannot remember arbitrary rows removed before parsing and later restored.

`assign`, `drop` and other native pandas/xarray operations handle adding or removing values. Selecting one stream does not silently mutate its companions; the trial-row slicing examples show the explicit connection to pose and events.

The figure notebooks loop over multiple selected recordings and keep their raw clocks separate. Calibration, pose preparation and background images belong to each recording. Only the resulting trial measurements are combined for comparison.

## Conversion and optional processing

The existing `data_conduit.integrations.DLC.pose.pose_to_movement` converts aligned raw DLC arrays to movement's singular `time`, `keypoint`, `individual` dimensions. It validates shared coordinates and acquired seconds. Raw DLC arrays retain their original names.

Camera timestamps retain acquisition order within each filename-ordered chunk. The catalog rejects invalid clocks before positional pose alignment; equal frame counts still assume corresponding source frames.

`prepare_pose` optionally masks low confidence, interpolates short internal gaps, then applies a median filter. The function defaults disable every option. The figure settings retain an explicit 0.9 confidence threshold from the original examples, with interpolation/smoothing disabled; inspect tracking quality before accepting that threshold. The data demo disables all processing by default. Choices are documented beside the calls. Processing returns a copy and runs on each full recording before trial slicing. The interpolation limit and smoothing window are frame counts, not elapsed-time thresholds.

## Figures

| Notebook | Measurement and output |
|---|---|
| Kinematics | Direct movement speed and head-direction calculations, then wrapped traces of preselected intervals |
| Spatial | Head direction relative to an external reference; bearings towards fixed or moving ROI positions; rectangular occupancy bins |
| Trajectories | Outbound path length versus mean inbound movement path deviation, paired by recording and trial |
| Timeseries template | Existing annotations drawn on actual movement speed for selected trials |

Kinematics and the timeseries demo calculate derivatives from acquired sample times. These calculations can span acquisition pauses; breaks in the plotted line do not remove those estimates. Optional interpolation/smoothing uses frame counts and remains disabled by default.

Pixel/centimetre display uses explicit measured calibration per recording. A scalar scale is isotropic: it is not a perspective or fisheye correction. Static/moving ROIs need actual positions in the same coordinate frame as tracking.

The existing `refactor_qc` training-session/minimum-trial/LED rules can be enabled before plotting. Whole-recording ON exclusion in the figures is a separate, explicitly named option. The LED field denotes ON-event presence within a parsed trial; it does not reconstruct persistent hardware state.

## Important interpretation limits

- The inbound summary is the sample mean of unsigned perpendicular deviations from the infinite line through the observed inbound endpoints. It is not home-port angular error.
- Missing or degenerate paths are reported rather than assigned plausible-looking measurements. Interpolation, if enabled, changes the positions used for those measurements.
- Occupancy bins are rectangles. Empty cells are transparent; no circular clipping of the bin geometry is performed.
- `read_session_video_frame` remains in `movement_figures.video`. The name `UndistortedVideoData` alone does not verify lens correction or agreement with pose coordinates.
- Function and source-reader checks used movement 0.17.0 and separate development fixtures. The uploaded ZIPs contain code; these checks are not a run of your experimental recordings.

For the manual repair, use the accompanying numbered edit guide. It compares the original uploads with these reference files and identifies edits that belong together. No installer or automatic notebook migration is included.




==================================================



================================================
# Movement figure notebooks

These notebooks load one recording, calculate movement measurements, and show how
to call the figure functions with explicit inputs. Functions return measurements,
figures, axes, and reports; the example cells control display with `plt.show()`.
Definitions and examples are separated, with numbered section comments inside the
longer functions.

| Notebook | Purpose |
|---|---|
| [Data import demo](data_template/demo.ipynb) | Configure and load a session; inspect coverage, coordinates, and pose processing |
| [Annotation demo](timeseries_template/demo.ipynb) | Self-contained synthetic examples of trial regions, events, outcomes, legends, and relative time |
| [Kinematics](Figures/Kinematics/kinematics.ipynb) | Separate x position, y position, linear speed, and angular head velocity figures, each wrapped into continuous time rows |
| [Spatial](Figures/Spatial/spatial.ipynb) | Allocentric and body-relative head angles; occupancy over the first matching video frame |
| [Trajectories](Figures/Trajectories/trajectories.ipynb) | Two final plotting options: metric explanations and trial summaries, or a QC-style outbound/inbound path grid |

[Which movement functions are used, and what is custom?](MOVEMENT_FUNCTIONS.md)
provides the API map, callable inputs/returns, and measurement definitions. Shared
session helpers live in [loading.py](data_template/loading.py); experiment-specific
annotations live in [annotations.py](timeseries_template/annotations.py).

## Run on your recording

Use the project's Python 3.12 kernel with this checkout and **movement 0.17.0**, the
installed version used to check the API calls. The loader requires movement
0.17.0 or newer; compatibility with later releases has not been established here.
Video backgrounds use `imageio`; DLC HDF5 input also needs PyTables.

Start Jupyter inside the checkout and run the data import demo first. `ROOT` is
the directory above the hierarchy, while `SESSION` is only the session folder
name. The default layout is `ROOT / mouseID / day / SESSION`; use
`LEVEL_NAMES = ()` for `ROOT / SESSION`. `LEVEL_SELECTORS` can disambiguate
repeated session names. The selection must identify exactly one recording.

`SOURCES = ("trials", "events", "dlc")` requests parsed trials, raw event markers,
and pose. Data-conduit reads the video timestamp stream as a DLC alignment
dependency. Loading a video image for the background is a separate call.

Run each figure notebook from top to bottom:

1. Set the session, individual, keypoint names, and pose-processing controls.
2. Call `load_figure_data(...)` and inspect the returned coverage and trial table.
3. Run the function-definition cells, then the measurement example.
4. Choose display controls and run the plotting examples. Returned figures can
   also be exported, for example `x_position_figure.savefig("x_position.png", dpi=300)`.

`FigureData` preserves raw and processed pose separately. Its `position` array
contains all keypoints for one animal, with dimensions `(time, space, keypoint)`;
`point` contains one selected landmark, with dimensions `(time, space)`. The
returned trial table is sorted by start time and has zero-based row labels.
`TRIAL_ROWS` refers to these rows, not a trial ID stored in the source data.

Pose processing runs on the continuous session in this order: confidence masking,
optional short-gap interpolation, then optional rolling median. The current
confidence threshold of 0.9 is an example setting. `MAX_GAP_FRAMES = None` and
`SMOOTHING_WINDOW = None` disable those operations. Sizes count samples, not seconds.
Changing session or processing controls requires rerunning loading and subsequent
cells; changing only plotting controls reuses the calculated measurements.

## Display controls

| Notebook | Controls and effect |
|---|---|
| Kinematics | `WINDOW` sets absolute display bounds and overrides `TRIAL_ROWS`. Otherwise selected rows define one continuous span, including intervening trials and gaps. `ROW_DURATION_S` wraps that span into equal-duration rows; the example uses 25 seconds. Set both selection controls to `None` for the complete recording. |
| Spatial angles | `PLOT_MODE` is `"scatter"`, `"line"`, or `"scatter+trace"`. The last option adds a circular rolling mean, not a fit. `TRACE_WINDOW_SAMPLES` sets its odd sample window; `TRACE_MAX_GAP_SECONDS` controls line/averaging breaks at clock gaps. `WINDOW` affects the angle display. |
| Spatial occupancy | `OCCUPANCY_BINS`, `ARENA_RANGE`, and `OCCUPANCY_ALPHA` control binning, optional crop, and transparency. `USE_OCCUPANCY_VMAX` switches between a manual `OCCUPANCY_VMAX` ceiling and the observed maximum. These controls affect the full-session/phase heatmaps independently of the angle window. |
| Trajectories, option 1 | `EXAMPLE_TRIAL_ROWS = None` selects and labels the first valid outbound/inbound examples. A dictionary such as `{"outbound": 0, "inbound": 0}` requests specific examples. The accompanying summary shows each eligible phase in trial order and counts exclusions. |
| Trajectories, option 2 | `PATH_GROUP_BY` selects outcome, session, or individual-trial rows. The example uses `TRIAL_ROWS` when grouping by trial. `PATH_CMAP`, `PATH_POINT_SIZE`, `PATH_ALPHA`, `BACKGROUND_ALPHA`, and `ARENA_RANGE` control appearance. Trial colours retain their full-session order. The secondary metric/count table covers the displayed trial subset. |

The background must match the image coordinates tracked by DLC.
`read_session_video_frame(..., video_subdir="UndistortedVideoData")` reads frame
zero of the first filename-ordered video chunk. For raw-video tracking, pass
`"VideoData"` instead. Spatial exposes this as `OCCUPANCY_VIDEO_SUBDIR`; the
trajectory loading cell uses the helper's undistorted-video default. No image
resizing, undistortion, or coordinate calibration happens in the plotting code.
For automatic occupancy bounds, a stationary x or y coordinate receives a
±0.5-pixel range so its histogram bins still have finite width.

## Units and interpretation

Positions and path deviations are **pixels**, speed is **pixels/second**, and
angles/angular velocity are **radians**/**radians per second**. Time remains the
recorded session clock. Image y increases downward, so positive head rotation is
clockwise. Occupancy counts **valid position samples per bin**; it is neither
dwell-time seconds nor a probability. A colour ceiling changes displayed colours,
not the retained counts.

Outbound straightness is endpoint distance divided by travelled path length,
**D/L**. Inbound deviation measures distance to the infinite line through the
observed inbound endpoints, rather than a specified home port. Phase metrics
exclude remaining missing coordinates; paths can still be shown when enough
observations remain, with their exclusion status reported separately. Missing
timestamps are a distinct issue from missing coordinate values.

## Validation status

The [permanent regression tests](tests/test_figure_functions.py) check
measurements, missing-data handling, circular means, wrapped timelines, occupancy
counts/colour limits, trajectory subset summaries and all-excluded displays,
annotation behaviour, and notebook compilation/docstrings.
From the `data-conduit/` directory, with the project environment active, run:

```bash
python -m unittest discover -s src/movement_figures/tests -v
```

These tests extract notebook function definitions and use synthetic inputs; they
need no recording drive and do not save synthetic outputs into the notebooks.
Additional development checks execute every notebook's example workflow with
synthetic inputs and produce labelled synthetic previews.

The three figure notebooks were executed successfully in fresh project kernels
on **2026-09-08**, using the reconnected recording
`MR_M01569515 / Day21 / 2026-06-21T093349Z`. Their real outputs are saved in the
notebooks: four kinematic figures, two spatial figures, and three trajectory
figures covering both final options. All were visually inspected.

The session contains **106 trials** and **36,354 aligned pose frames**; **35,654**
body frames remain valid with confidence threshold 0.9 and interpolation/smoothing
disabled. Strict path eligibility leaves **29 outbound** and **60 inbound**
measurements; the remaining 77/46 phases contain unfilled tracking gaps. These
exclusions are shown explicitly rather than silently replaced with measurements.
The occupancy background is the first matching undistorted video frame
(1620 × 1260 pixels), with a shared display ceiling of 100 samples/bin.

The annotation demo intentionally uses synthetic data. The data import demo
retains its placeholder configuration. Execution and layout checks do not by
themselves establish tracking quality or the suitability of processing settings
for an experimental analysis.

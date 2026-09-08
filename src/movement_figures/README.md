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

# Movement APIs and the code added around them

The tables below describe the calls in these notebooks and shared helpers,
checked against the installed **movement 0.17.0** source. They list APIs called
directly; movement may call other functions internally. Notebook functions have
explicit arguments and return values, and their full docstrings are beside the
implementation.

## Where movement is used

| Notebook/helper | Direct movement calls | Output used here |
|---|---|---|
| [Loading](data_template/loading.py): `_validate_pose` | `movement.validators.datasets.ValidPosesInputs.validate` | Validates the pose schema; the helper additionally checks a finite, increasing acquisition clock |
| Loading: `prepare_pose`, called by `load_figure_data` | `movement.filtering.filter_by_confidence`, `interpolate_over_time`, `rolling_filter` | A processed position copy: confidence mask, optional linear gap filling, optional median smoothing |
| [Kinematics](Figures/Kinematics/kinematics.ipynb): `compute_kinematic_measurements` | `movement.kinematics.compute_speed`, `compute_head_direction_vector`; `movement.utils.vector.compute_signed_angle_2d` | Linear speed and head vectors/rotations; the notebook adds angular change divided by elapsed time |
| [Spatial](Figures/Spatial/spatial.ipynb): `compute_head_angles` | `movement.kinematics.compute_forward_vector_angle`, `compute_head_direction_vector`; `movement.utils.vector.compute_signed_angle_2d` | Allocentric head heading and head angle relative to the body axis |
| Spatial: `plot_spatial_occupancy` | `movement.plots.plot_occupancy` | Raw 2-D sample counts `h`, `xedges`, `yedges`, plus a plotted histogram |
| [Trajectories](Figures/Trajectories/trajectories.ipynb): `measure_trial_paths` | `movement.kinematics.compute_path_straightness`, `compute_path_length`, `compute_path_deviation` | Outbound D/L and length; per-sample inbound perpendicular deviation |
| Trajectories: `plot_session_path_grid` | `movement.plots.plot_centroid_trajectory` | Observed x/y samples for a single configured landmark, coloured by trial order |
| Trajectories: `summarise_path_metrics` | None; pandas groups the existing results | One metric name, units, valid/excluded counts, and eligible median for each phase |
| [Data import demo](data_template/demo.ipynb) | The loading/processing calls above | Session previews, schema summaries, and raw/processed missing-coordinate counts |
| [Annotation demo](timeseries_template/demo.ipynb) and [annotations.py](timeseries_template/annotations.py) | None | Custom experiment annotations and Matplotlib figure layouts |

The installed `movement.plots` module exposes `plot_centroid_trajectory` and
`plot_occupancy`; both are used above. Kinematic timelines, head-angle time series,
metric geometry, and trial-summary figures are drawn with Matplotlib. The
custom figures reuse movement measurements without claiming those layouts are
movement plotting APIs.

## Shared loading inputs and returns

```python
config = build_config(
    root,
    session=session_folder_name,
    level_names=("mouseID", "day"),
    level_selectors=None,
    sources=("trials", "events", "dlc"),
)
data = load_figure_data(
    config,
    individual="individual_0",
    tracking_keypoint="body",
    confidence_threshold=None,
    max_gap_frames=None,
    smoothing_window=None,
)
frame, video_path = read_session_video_frame(
    session_path, video_subdir="UndistortedVideoData"
)
```

`load_figure_data` returns `FigureData` with `session_data`, `raw_pose`, `pose`,
`position`, `point`, `trials`, and `events`. `position` has dimensions
`(time, space, keypoint)`; `point` has `(time, space)`. Trials are sorted and their
index reset to zero-based rows. `read_session_video_frame` returns an RGB NumPy
array `(height, width, 3)` and its source path. Neither function displays figures.

`prepare_pose(raw_pose, *, confidence_threshold=None, max_gap_frames=None,
smoothing_window=None)` returns a separately processed pose Dataset. Each `None`
skips that operation. When enabled, calls occur in this order:

1. `filter_by_confidence(position, confidence, threshold=...)` masks positions
   below the threshold; it retains the measured confidence values.
2. `interpolate_over_time(position, max_gap=...)` fills at most that many
   consecutive NaN samples using movement's default linear method. The installed
   implementation interpolates by sample order, not the spacing of timestamps.
3. `rolling_filter(position, window=..., statistic="median")` uses a sample
   window and the default full-window nonmissing requirement.

The helper requires an odd median window of at least three samples. These pose
operations run before display windows or trial phases are selected.

## Callable figure workflow

All plotting functions return their figure and axes. Call `plt.show()` in the
example cell, or call `savefig(...)` on the returned figure. Selection and styling
arguments do not silently replace the notebook's input arrays.

| Function | Main inputs and controls | Returns |
|---|---|---|
| `compute_kinematic_measurements` | `position`; required `tracking_keypoint`, `left_ear`, `right_ear`; optional `camera_view` | Dataset: `x_position`, `y_position`, `linear_speed`, `angular_head_velocity` |
| `select_kinematic_window` | `time`, `trials`; `trial_rows=(0, 1, 2)`, `window=None` | Absolute `(start, end)`, clipped to the recording; an explicit window overrides rows |
| `plot_wrapped_kinematic_trace` | One `trace`, `trials`, `events`; required `window`, `row_duration_s`, `ylabel`, `title`; `y_mode`, `y_limits`, `max_gap_s`, `color` | `(figure, axes)` with a one-dimensional row-axes array |
| `compute_head_angles` | `position`; ear and body keypoint names; `reference_vector`, `camera_view` | Dataset: `allocentric_heading`, `head_relative_to_body`, in radians |
| `circular_rolling_trace` | One `angle`; `window_samples=15`, `max_gap_seconds=None` | Circular mean DataArray on the original clock |
| `plot_head_angles` | `angles`, `trials`, optional `events`; `window`, `mode`, `trace_window_samples`, `max_gap_seconds` | `(figure, axes, traces)`; traces is a full-clock Dataset only in `"scatter+trace"` mode, otherwise `None` |
| `compute_phase_masks` | `time`, `trials` | `(phase_masks, valid_phase_trial_count)`; masks have one Boolean per input time |
| `plot_spatial_occupancy` | `point`, `phase_masks`, `background_frame`; `bins`, `arena_range`, `use_vmax`, `vmax`, `alpha`, `title` | `(figure, axes, occupancy)`; occupancy maps nonempty panel labels to raw histogram dictionaries |
| `measure_trial_paths` | `point`, `trials` | `(metrics, phase_paths, inbound_deviations)`; paths are keyed by `(trial_row, phase)`, deviations by valid inbound trial row |
| `draw_arena` | `ax`, `frame`; `arena_range`, `background_alpha`, `show_ticks` | The configured axis |
| `plot_session_path_grid` | `phase_paths`, `metrics`, `frame`; `group_by`, `trial_rows`, labels, crop, colour and opacity controls | `(figure, axes, plot_report)`; report includes each phase's metric status and `path_shown` flag |
| `plot_metric_geometry` | `phase_paths`, `metrics`, `inbound_deviations`, `frame`; optional phase-to-row mapping `trial_rows`, label/crop/opacity | `(figure, axes, example_report)` identifying the chosen examples and their eligibility |
| `plot_trajectory_summary` | `metrics`; `session_label`, optional `outcome_colors` | `(figure, axes)` showing eligible measurements in trial order and exclusion counts |
| `summarise_path_metrics` | `metrics`, or a selected `path_report` from the grid | Phase-indexed DataFrame: `metric`, `units`, `n_trials`, `n_valid`, `n_excluded`, `median`; the median stays NaN when no phase is eligible |

In Trajectories, the two final options reuse the same `measure_trial_paths`
results. **Option 1** shows the metric geometry and a trial-order summary;
**option 2** shows the outbound/inbound arena grid and calls
`summarise_path_metrics(path_report)` for a compact table covering exactly the
trials displayed. Each phase has one named metric and units, a median of eligible
values, and total/valid/excluded counts. An all-excluded phase keeps a NaN median.
`plot_session_path_grid(group_by="outcome")` uses outcome rows;
`"session"` uses one row, and `"trial"` uses individual trial rows. The explicit
example passes `TRIAL_ROWS` only for the trial grouping. The function itself can
filter `trial_rows` with any grouping.

## What the surrounding code adds

**Recording discovery, trials, and a shared clock.** Data-conduit selects the
single session, parses experiment events into trials, reads DLC files, and aligns
DLC frame positions with the recorded video timestamps. The adapter converts the
arrays into movement's singular dimension names and required axis order. This
workflow uses existing data-conduit readers rather than movement's file-loading
APIs. The figure loader also preserves raw pose, exposes one individual/landmark,
and provides the sorted trial table. Raw `Start Trial` events remain distinct
from derived trial boundaries.

**Angular head velocity from movement primitives.** movement 0.17.0 has no
dedicated angular-head-velocity function in its public kinematics API. The
notebook takes the signed rotation from the previous head vector to the current
one and divides by that interval's recorded duration. Each result is the average
angular velocity over the preceding interval; the first sample and intervals
involving missing head positions are NaN. The signed-angle calculation handles
the ±π heading wrap, assuming less than half a turn between adjacent samples.

**Explicit phase boundaries and metric eligibility.** A valid target trigger
splits a trial into outbound and inbound. Occupancy uses half-open intervals
`[start, target)` and `[target, end)`: a target sample counts as inbound, and the
trial-end sample is excluded. Path metrics include both phase endpoints, so an
exact target sample belongs to both paths as their shared endpoint. The notebook
requires full pose-clock coverage, at least two samples, and finite coordinates
throughout a phase before measuring it. This guards against the default
path-length forward filling of remaining NaNs. It does not detect missing
timestamps. A partially observed path can still appear in the grid when it has
at least two finite x/y samples; `path_shown` and metric eligibility are reported
separately.

**Circular display summaries and gap-aware lines.** Spatial's
`"scatter+trace"` mode overlays a centred rolling mean of sine and cosine,
converted back with `atan2`. It is a descriptive circular mean, not a fitted
trend, regression, or prediction. Averaging stops at missing samples and large
clock gaps; shorter windows are used at segment ends, and means with near-zero
resultant length become NaN. Drawn angle lines also break at the ±π boundary.
`"line"` shows the measured angles with those drawing breaks and no averaging.
This is custom display logic using NumPy/pandas; the pose median above still
uses movement. No claim is made that movement lacks every related filtering API.

**Wrapped kinematic figures.** Four separate figures each follow one continuous
session-time interval, retaining inter-trial time. `ROW_DURATION_S` controls
wrapping, not trial selection or derivative calculation. Each row has the same
time width and y limits; unused space after the selected interval is shaded.
`max_gap_s=None` breaks drawn lines at gaps greater than five times the median
frame interval. Spatial's analogous default is three times the median interval.
These display thresholds do not repair timestamps or change measurements.

**Camera backgrounds and shared colour controls.** The video helper reads frame
zero from the first filename-ordered chunk. Matplotlib places pixel centres at
integer coordinates using half-pixel image edges, with y increasing downward.
The video must already match the raw/undistorted coordinates used for DLC.
Occupancy panels use common spatial bounds, bin counts, and one colour scale.
Raw histogram indexing is `[x_bin, y_bin]`; its transpose is used for the plotted
mesh. Empty bins are transparent. A manual colour maximum caps displayed colour
only; counts above it retain their values and use the top colour. The shared
colourbar indicates an exceeded ceiling. Raising the ceiling reduces colour
saturation. `ARENA_RANGE=None` bins over finite tracked coordinates while showing
the complete image. If all finite samples share one x or y value, the inferred
range on that axis expands by ±0.5 pixels to give finite-width bins. Explicit
bounds must increase on both axes and also crop the display.

**Experiment annotations, explanatory geometry, and summaries.** The custom
annotation module supplies outbound/inbound spans, raw events, outcome symbols,
and shared legends. A Miss has a trial-end marker; a decision-poke marker also
requires a valid chosen port. Trajectory layouts select/group trials, colour
paths by their full-session order, and count exclusions. The metric illustration
computes endpoint distance and geometric projections to explain the measurement;
reported D/L, length, and deviations come from movement. With
`EXAMPLE_TRIAL_ROWS=None`, each geometry panel labels its first valid phase; it is
an example, not an automatically representative trial. Arithmetic per-sample
means and eligible-trial medians/counts are explicit xarray/pandas summaries.
`summarise_path_metrics` uses only rows marked `status == "ok"` for its medians,
keeps outbound/inbound metrics separate, and follows the input table's trial scope.

## Physical meanings and limits

- Positions, endpoint distances, path lengths, and deviations are **pixels**.
  There is no pixel-to-distance calibration. Speed is **px/s**; head angles and
  angular velocity are **rad** and **rad/s**. The default image axes make
  positive rotation clockwise.
- Allocentric heading is relative to `REFERENCE_VECTOR`, which defaults to
  image-right. Body-relative head direction compares the head with the
  `BODY_BACK → BODY_FRONT` axis. It does not measure a target's bearing.
- Outbound straightness is **D/L**: endpoint distance divided by travelled path
  length. One means a direct straight path; smaller values mean less directness.
  It does not quantify every meaning of path complexity. Zero-length paths are
  undefined. Sampling, jitter, and pose processing affect path length.
- Inbound deviation is the unsigned perpendicular distance to the **infinite
  line through the observed inbound endpoints**. It is not error relative to a
  configured home port. Coincident endpoints cannot define that line. The
  reported mean weights recorded samples equally rather than elapsed time.
- Occupancy is **valid position samples per spatial bin**, with no time or
  probability normalization. Unequal frame spacing and different phase durations
  affect counts. Whole-session occupancy includes inter-trial periods and trials
  whose phase boundaries are unavailable.

The three figure notebooks were also executed on the reconnected real recording
on 2026-09-08, with their outputs saved and visually inspected. See the
[README](README.md#validation-status) for session details and exclusion counts.
Earlier synthetic previews remain separate from these recording results.

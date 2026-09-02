# Pose Trajectory Analysis

A `movement`-driven workflow for the path-integration task: it segments each trial into
**inbound** and **outbound** paths, builds a per-trial reference table, and computes
trajectory-complexity, general behaviour, and reward-relative measures, all natively in the
[`movement`](https://movement.neuroinformatics.dev/) library.

It is also a teaching artifact. The design rule is deliberate:

- **The notebooks do all the analysis.** Every `movement` call and every plot is written out,
  in the open, in a notebook, so the notebooks read as a clear demonstration of how to use
  `movement`.
- **The package (`pose_trajectory/`) only pulls and formats data** so it can be fed into
  `movement`: selecting sessions, reading DeepLabCut pose and events, parsing trials, computing
  segment windows, and inferring nosepoke positions. No package function hides a `movement`
  computation.

---

## Environment

Run everything with the repo's **`.venv`** (Python 3.12), which has `data_conduit` (editable),
`movement` (from the project's `main` branch, for the native path metrics), `harp-python`,
`tables`, `matplotlib`, `seaborn`, `itables`, and `ipykernel`. Select `.venv` as the notebook
kernel.

> The trajectory metrics `compute_path_straightness`, `compute_directional_change`, and
> `compute_path_deviation` exist only on `movement`'s `main` branch (not in the 0.15.0 release),
> which is why `.venv` installs `movement` from `main`. `main` requires Python >= 3.12.

---

## Data layout

The workflow expects one mouse folder containing day folders, each containing session folders:

```
<DATA_ROOT>/<MOUSE_ID>/<day>/<session>/{ExperimentEvents, VideoData, DLC, SessionSettings, ...}
```

For future data organised as `<DATA_ROOT>/<PHASE>/<MOUSE_ID>/<day>/<session>`, simply include
the phase in `DATA_ROOT`. Point each notebook's `EDIT THESE` config block at the mouse folder.

---

## How to run

Open the notebooks in order and run top to bottom. Each has an `EDIT THESE` config block; keep
it identical across notebooks so they find the same sessions. Notebook 01 must run first (it
saves the table the others read); notebook 05 also needs notebook 04.

| Notebook | What it does | Plan step |
|---|---|---|
| [`01_summary_table.ipynb`](notebooks/01_summary_table.ipynb) | Build the catalog from scratch, select sessions, combine them, and build + save the per-trial **summary / reference table** (with inbound/outbound segment times). | 0, 1, 2 |
| [`02_trajectory_complexity.ipynb`](notebooks/02_trajectory_complexity.ipynb) | `movement` path metrics (length, straightness, directional change, deviation) on the body centroid, per segment, distributed by outcome. | 3 |
| [`03_behaviour_over_time.ipynb`](notebooks/03_behaviour_over_time.ipynb) | Head direction, angular head velocity, speed, displacement: example traces + per-trial summaries by outcome. | 4 |
| [`04_infer_poke_positions.ipynb`](notebooks/04_infer_poke_positions.ipynb) | Infer each nosepoke's pixel position **per session** from pose at poke time, with an accuracy report and a ring visualisation. | (prep for 5) |
| [`05_reward_relative.ipynb`](notebooks/05_reward_relative.ipynb) | Distance, head direction, and angular velocity **relative to the rewarded port**, per segment, by outcome. | 5 |

Tables are written to `outputs/`; figures to `figures/`.

---

## Package modules (`pose_trajectory/`)

| Module | Role |
|---|---|
| [`catalog.py`](pose_trajectory/catalog.py) | `build_catalog()`: the per-session extraction spec, assembled from scratch and fully commented (events, nosepoke, soundcard, session_settings, video, dlc). |
| [`loading.py`](pose_trajectory/loading.py) | `list_sessions`, `build_session_group`, `combine_all` (the cross-session container), and `session_movement_dataset` (one session's pose, formatted for `movement`). |
| [`trials.py`](pose_trajectory/trials.py) | Re-exports the lab's canonical `parse_trials`; `build_summary_table` adds the inbound/outbound segment windows. |
| [`infer_poke_positions.py`](pose_trajectory/infer_poke_positions.py) | Per-session nosepoke position inference + accuracy/spread report + inter-port distances. |
| [`workspace.py`](pose_trajectory/workspace.py) | Output/figure directories, `save_figure`, `day_order`. |

---

## Definitions

**Trial segments** (every trial has both legs):

- `outbound_start` = the previous trial's poke (or, for the first trial of a session, its
  Start-Trial event)
- `outbound_end` = `inbound_start` = **target zone triggered**
- `inbound_end` = **poke**

**Head direction**: `movement`'s `compute_forward_vector_angle` on the two ear keypoints
(`lear`, `rear`), top-down camera.

**Centroid**: trajectory metrics use the mean of all keypoints (the body centroid).

---

## Notes

- **Reward inference (notebook 04/05) is exploratory.** Positions come from the head at
  Success/Fail poke times, so in a well-trained session where most pokes hit one port, only a
  few ports are resolved. Sessions with more errors, or where the rewarded port varies, populate
  more of the ring; the richer alternative source is the raw `nosepoke:Activations` stream.
- **`movement` schema.** `movement` (main) uses singular `keypoint` / `individual` dimensions;
  `session_movement_dataset` renames `data_conduit`'s plural names to match.
- **Reuse.** Trial parsing reuses `q_c_data_analysis/firstdata`'s `parse_trials`; the
  inbound/outbound definitions follow that project's `trajectory_metrics/segmentation.py`.

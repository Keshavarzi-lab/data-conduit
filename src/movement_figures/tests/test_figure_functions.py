"""Regression checks for notebook functions; no recording drive is required.

Run with the project's movement environment:
    python -m unittest discover -s src/movement_figures/tests -v

Only imports and function definitions are extracted from notebooks. Their real
session example calls are never run, and no synthetic output is saved into them.
"""

import ast
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

FIGURE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURE_ROOT.parent))


def notebook_functions(relative_path):
    """Load the real function definitions without executing example cells."""
    path = FIGURE_ROOT / relative_path
    context = {"__name__": "notebook_function_test"}
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        definitions = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        ]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), context)
    return context


def point_array(values, times=None):
    """Build a small labelled x/y path on an explicit clock."""
    values = np.asarray(values, dtype=float)
    times = np.arange(len(values), dtype=float) if times is None else times
    return xr.DataArray(values, dims=("time", "space"), coords={"time": times, "space": ["x", "y"]})


class FigureFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = notebook_functions("Figures/Kinematics/kinematics.ipynb")
        cls.spatial = notebook_functions("Figures/Spatial/spatial.ipynb")
        cls.trajectories = notebook_functions("Figures/Trajectories/trajectories.ipynb")
        cls.annotations = notebook_functions("timeseries_template/demo.ipynb")
        cls.data = notebook_functions("data_template/demo.ipynb")

    def tearDown(self):
        plt.close("all")

    def test_constant_speed_and_missing_ear_measurements(self):
        times = np.arange(100.0, 110.0, 0.5)
        body = np.column_stack([3 * (times - 100), 4 * (times - 100)])
        positions = np.stack([body, body + [0, -1], body + [0, 1]], axis=2)
        position = xr.DataArray(
            positions, dims=("time", "space", "keypoint"),
            coords={"time": times, "space": ["x", "y"], "keypoint": ["body", "lear", "rear"]},
        )
        position.loc[{"time": times[5], "keypoint": "lear"}] = np.nan
        measurements = self.kinematics["compute_kinematic_measurements"](
            position, tracking_keypoint="body", left_ear="lear", right_ear="rear",
        )
        np.testing.assert_allclose(measurements.linear_speed, 5.0)
        self.assertTrue(np.isnan(measurements.angular_head_velocity.values[[0, 5, 6]]).all())
        np.testing.assert_allclose(measurements.angular_head_velocity.dropna("time"), 0, atol=1e-12)
        np.testing.assert_array_equal(measurements.time, times)
        self.assertEqual(measurements.angular_head_velocity.attrs["units"], "rad/s")

    def test_wrapped_rows_follow_time_not_trial_boundaries(self):
        time = np.arange(100.0, 151.0)
        trace = xr.DataArray(np.sin(time), dims="time", coords={"time": time})
        trials = pd.DataFrame({
            "start_time": [102.0, 130.0], "end_time": [115.0, 149.0],
            "tz_triggered_time": [110.0, 145.0], "outcome": ["Success", "Miss"],
            "ChosenPort": [1, -1],
        })
        window = self.kinematics["select_kinematic_window"](trace.time, trials, trial_rows=(0, 1))
        self.assertEqual(window, (102.0, 149.0))
        with patch.object(plt, "show", side_effect=AssertionError("Plot functions must return figures.")):
            fig, axes = self.kinematics["plot_wrapped_kinematic_trace"](
                trace, trials, None, window=window, row_duration_s=20,
                ylabel="Synthetic value", title="Synthetic wrapped timeline",
            )
        np.testing.assert_allclose([ax.get_xlim() for ax in axes], [(102, 122), (122, 142), (142, 162)])
        np.testing.assert_allclose([ax.get_ylim() for ax in axes], [axes[0].get_ylim()] * 3)
        self.assertEqual(len(fig.legends), 1)
        self.assertTrue(all(ax.get_legend() is None for ax in axes))
        fig.canvas.draw()

        # Floating-point rounding must not create an empty fourth row.
        fractional = xr.DataArray([0., 1., 2., 3.], dims="time", coords={"time": [.1, .2, .3, .4]})
        _, fractional_axes = self.kinematics["plot_wrapped_kinematic_trace"](
            fractional, trials.iloc[:0], None, window=(.1, .4), row_duration_s=.1,
            ylabel="Synthetic value", title="Fractional row durations",
        )
        self.assertEqual(len(fractional_axes), 3)

    def test_circular_trace_never_averages_wrap_to_zero_or_fills_gaps(self):
        angle = xr.DataArray(
            np.deg2rad([179, -179, 178, np.nan, 90, 90, -90, -90]),
            dims="time", coords={"time": [0., 1., 2., 3., 4., 5., 20., 21.]},
        )
        trace = self.spatial["circular_rolling_trace"](angle, window_samples=3)
        self.assertTrue((np.abs(trace.values[:3]) > 3.1).all())
        self.assertTrue(np.isnan(trace.values[3]))
        np.testing.assert_allclose(trace.values[4:], [np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2])
        segments = self.spatial["angle_segments"](angle, split_wrap=True)
        self.assertEqual([segment.tolist() for segment in segments], [[0], [1], [2], [4, 5], [6, 7]])

    def test_occupancy_phase_counts_and_display_cap_are_independent(self):
        point = point_array([[10, 10], [10, 10], [20, 20], [20, 20], [10, 10],
                             [10, 10], [20, 20], [20, 20], [30, 30]])
        trials = pd.DataFrame({"start_time": [0., 4.], "tz_triggered_time": [2., 6.], "end_time": [4., 8.]})
        masks, count = self.spatial["compute_phase_masks"](point.time, trials)
        self.assertEqual(count, 2)
        self.assertEqual(np.flatnonzero(masks["Outbound"]).tolist(), [0, 1, 4, 5])
        self.assertEqual(np.flatnonzero(masks["Inbound"]).tolist(), [2, 3, 6, 7])
        frame = np.full((40, 40, 3), 128, dtype=np.uint8)
        results = []
        for use_vmax in (True, False):
            fig, axes, occupancy = self.spatial["plot_spatial_occupancy"](
                point, masks, frame, bins=2, arena_range=((0, 40), (0, 40)),
                use_vmax=use_vmax, vmax=1.0, title="Synthetic occupancy",
            )
            self.assertEqual(occupancy["Whole session"]["h"].sum(), 9)
            self.assertEqual(occupancy["Outbound"]["h"].sum(), 4)
            self.assertEqual(occupancy["Inbound"]["h"].sum(), 4)
            self.assertEqual(len(fig.axes), 4)
            self.assertEqual(axes[0].images[0].get_extent(), [-0.5, 39.5, 39.5, -0.5])
            expected_max = 1.0 if use_vmax else max(info["h"].max() for info in occupancy.values())
            self.assertTrue(all(ax.collections[0].norm.vmax == expected_max for ax in axes))
            results.append(occupancy)
            fig.canvas.draw()
        for label in results[0]:
            np.testing.assert_array_equal(results[0][label]["h"], results[1][label]["h"])

        # A stationary animal still occupies a real bin on each spatial axis.
        _, _, stationary = self.spatial["plot_spatial_occupancy"](point * 0 + 10, masks, frame, bins=2)
        self.assertEqual(stationary["Whole session"]["h"].sum(), 9)
        np.testing.assert_allclose(stationary["Whole session"]["xedges"][[0, -1]], [9.5, 10.5])
        np.testing.assert_allclose(stationary["Whole session"]["yedges"][[0, -1]], [9.5, 10.5])

    def test_path_metrics_have_known_geometry_and_exclude_missing_positions(self):
        point = point_array([[10, 10], [20, 20], [30, 10], [25, 15], [20, 10], [15, 15], [10, 10]])
        trials = pd.DataFrame({
            "start_time": [0., 4.], "tz_triggered_time": [2., 5.],
            "end_time": [4., 6.], "outcome": ["Success", "Failure"],
        })
        metrics, paths, deviations = self.trajectories["measure_trial_paths"](point, trials)
        first = metrics[(metrics.trial_row == 0) & (metrics.phase == "outbound")].iloc[0]
        self.assertAlmostEqual(first.path_straightness, 1 / np.sqrt(2))
        self.assertAlmostEqual(first.path_length_px, 20 * np.sqrt(2))
        self.assertAlmostEqual(first.endpoint_distance_px, 20)
        inbound = metrics[(metrics.trial_row == 0) & (metrics.phase == "inbound")].iloc[0]
        self.assertAlmostEqual(inbound.mean_deviation_px, 5 / 3)
        np.testing.assert_allclose(deviations[0], [0, 5, 0])
        self.assertEqual(float(paths[(0, "outbound")].time[-1]), float(paths[(0, "inbound")].time[0]))
        gapped = point.copy(deep=True)
        gapped.loc[{"time": 1.0}] = np.nan
        excluded, _, _ = self.trajectories["measure_trial_paths"](gapped, trials)
        row = excluded[(excluded.trial_row == 0) & (excluded.phase == "outbound")].iloc[0]
        self.assertEqual(row.status, "unfilled tracking gaps")
        self.assertTrue(np.isnan(row.path_straightness))
        self.assertTrue(np.isnan(gapped.sel(time=1)).all())

    def test_annotation_functions_return_figures_and_leave_inputs_intact(self):
        trials, events, time, values = self.annotations["make_annotation_example"](start_time_s=100)
        trial_copy = trials.copy(deep=True)
        annotations = self.annotations["qc_annotations"](trials, events)
        with patch.object(plt, "show", side_effect=AssertionError("Unexpected display side effect.")):
            fig, ax, result = self.annotations["plot_annotated_trace"](
                time, values, annotations, window=(99, 131), time_offset=100,
            )
            panels, axes, results = self.annotations["plot_annotation_panels"](
                time, {"one": values}, annotations, window=(100, 130), time_offset=100,
            )
        self.assertEqual(ax.get_xlim(), (-1, 31))
        self.assertEqual(axes.shape, (1,))
        self.assertEqual(len(results), 1)
        self.assertTrue(result.notes)
        pd.testing.assert_frame_equal(trials, trial_copy)
        fig.canvas.draw()
        panels.canvas.draw()

    def test_trajectory_options_explain_exclusions_and_summarise_displayed_trials(self):
        point = point_array([[10, 10], [20, 20], [30, 10], [25, 15],
                             [20, 10], [15, 15], [10, 10]])
        trials = pd.DataFrame({
            "start_time": [0., 4.], "tz_triggered_time": [2., pd.NA],
            "end_time": [4., 6.], "outcome": ["Success", "Miss"],
        })
        metrics, paths, deviations = self.trajectories["measure_trial_paths"](point, trials)
        frame = np.full((40, 40, 3), 128, dtype=np.uint8)
        grid, _, report = self.trajectories["plot_session_path_grid"](
            paths, metrics, frame, group_by="session", trial_rows=[1],
        )
        summary = self.trajectories["summarise_path_metrics"](report)
        self.assertEqual(summary.n_trials.tolist(), [1, 1])
        self.assertEqual(summary.n_valid.tolist(), [0, 0])
        self.assertEqual(summary.n_excluded.tolist(), [1, 1])
        self.assertTrue(summary["median"].isna().all())
        self.assertFalse(report.path_shown.any())
        eligible_summary = self.trajectories["summarise_path_metrics"](metrics)
        self.assertAlmostEqual(eligible_summary.loc["outbound", "median"], 1 / np.sqrt(2))
        self.assertEqual(eligible_summary.loc["inbound", "units"], "px")

        # Both final options remain explicit when every selected phase is excluded.
        excluded = metrics[metrics.trial_row == 1]
        geometry, axes, examples = self.trajectories["plot_metric_geometry"](
            paths, excluded, deviations, frame,
        )
        self.assertTrue((examples.valid_phases == 0).all())
        self.assertTrue(all(any("Metric excluded" in text.get_text() for text in ax.texts) for ax in axes))
        chronology, _ = self.trajectories["plot_trajectory_summary"](excluded)
        for fig in (grid, geometry, chronology):
            fig.canvas.draw()

    def test_all_notebooks_compile_and_every_function_has_documentation(self):
        import nbformat

        for path in FIGURE_ROOT.rglob("*.ipynb"):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                compile(cell.source, f"{path}:cell-{index}", "exec")
                for node in ast.walk(ast.parse(cell.source)):
                    if isinstance(node, ast.FunctionDef):
                        self.assertTrue(ast.get_docstring(node), f"Missing docstring: {path} {node.name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

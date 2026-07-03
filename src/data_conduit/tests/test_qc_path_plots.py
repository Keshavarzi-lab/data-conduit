import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_conduit.qc import path_plots


def _trial_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'mouseID': ['mouse_a', 'mouse_b'],
            'session': ['session_1', 'session_1'],
            'trial_index': [1, 2],
        }
    )


def test_grouped_plots_default_to_group_colour_ramp(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        path_plots,
        '_first_frame_for_session',
        lambda *args, **kwargs: np.zeros((4, 5, 3), dtype=np.uint8),
    )

    def fake_plot_paths(ax, pose, cell_trials, **kwargs):
        captured['color_column'] = kwargs['color_column']
        captured['color_keys'] = list(kwargs['color_cmaps'])
        captured['ramp'] = cell_trials[['mouseID', 'group', 'trial_index', '_ramp']].sort_values('trial_index').reset_index(drop=True)
        return cell_trials

    monkeypatch.setattr(path_plots, '_plot_paths_on_ax', fake_plot_paths)

    fig, axes = path_plots.plot_path_grid(
        {'trials': _trial_rows(), 'dlc:position': object()},
        '/tmp',
        groups={'g1': {'mouse_a', 'mouse_b'}},
        row_by='session',
        col_by='group',
        colorbar=False,
    )
    try:
        assert captured['color_column'] == 'group'
        assert captured['color_keys'] == ['g1']
        assert captured['ramp']['_ramp'].tolist() == [0.0, 1.0]
        assert axes[0][0].texts[0].get_text() == 'g1  n=2'
    finally:
        plt.close(fig)


def test_grouped_plots_can_still_colour_per_mouse(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        path_plots,
        '_first_frame_for_session',
        lambda *args, **kwargs: np.zeros((4, 5, 3), dtype=np.uint8),
    )

    def fake_plot_paths(ax, pose, cell_trials, **kwargs):
        captured['color_column'] = kwargs['color_column']
        captured['color_keys'] = list(kwargs['color_cmaps'])
        captured['ramp'] = cell_trials[['mouseID', 'group', 'trial_index', '_ramp']].sort_values('trial_index').reset_index(drop=True)
        return cell_trials

    monkeypatch.setattr(path_plots, '_plot_paths_on_ax', fake_plot_paths)

    fig, axes = path_plots.plot_path_grid(
        {'trials': _trial_rows(), 'dlc:position': object()},
        '/tmp',
        groups={'g1': {'mouse_a', 'mouse_b'}},
        row_by='session',
        col_by='group',
        color_by='mouseID',
        colorbar=False,
    )
    try:
        assert captured['color_column'] == 'mouseID'
        assert set(captured['color_keys']) == {'mouse_a', 'mouse_b'}
        assert captured['ramp']['_ramp'].tolist() == [0.0, 0.0]
        assert axes[0][0].texts[0].get_text().splitlines() == ['mouse_a  n=1', 'mouse_b  n=1']
    finally:
        plt.close(fig)
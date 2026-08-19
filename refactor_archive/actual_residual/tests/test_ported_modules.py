"""Regression checks for behavior merged while porting the remaining modules."""

import numpy as np
import pandas as pd
import xarray as xr

from data_conduit.actual.core.globaltimes.globaltimes_core import _match_idx
from data_conduit.actual.core.utils import _matches_selector, _parse_selectors
from data_conduit.actual.datasources.monosource import MonoSource
from data_conduit.actual.integrations.DLC.pose import align_pose_to_video


def test_global_time_exact_matching_handles_out_of_range_and_empty_targets() -> None:
    """Exact matching should use ``-1`` instead of indexing beyond the target."""
    np.testing.assert_array_equal(
        _match_idx(np.array([-1.0, 1.0, 4.0]), np.array([1.0, 2.0]), match_type="exact"),
        [-1, 0, -1],
    )
    np.testing.assert_array_equal(
        _match_idx(np.array([1.0, 2.0]), np.array([]), match_type="nearest"),
        [-1, -1],
    )


def test_selectors_accept_collections_and_reject_negative_levels() -> None:
    """The port should retain the live selector extensions."""
    assert _matches_selector("mouse_a", ("mouse_a", "mouse_b"))
    assert _matches_selector("mouse_a", {"mouse_a", "mouse_b"})

    try:
        _parse_selectors({"l-1_selector": "mouse_a"})
    except ValueError as error:
        assert "non-negative" in str(error)
    else:  # pragma: no cover - makes a missing validation failure explicit
        raise AssertionError("negative selector levels must be rejected")


def test_monosource_can_build_named_arrays_from_preloaded_frames() -> None:
    """A preloaded datasource should avoid filesystem I/O and expose its array."""
    frame = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.Index([0.0, 1.0], name="Time"),
    )
    source = MonoSource(
        dfs_dict={"events": frame},
        monosource_data_arrays={"events": {"l0_selector": "events"}},
        verbose=True,
    )

    assert list(source.data_arrays) == ["events"]
    assert source.data_arrays["events"].sizes["Time"] == 2


def test_dlc_pose_alignment_uses_video_frame_times() -> None:
    """Frame-indexed position/confidence arrays should receive the video clock."""
    position = xr.DataArray(
        np.zeros((2, 1, 2)),
        dims=("frame", "keypoints", "space"),
        coords={"frame": [0, 1], "keypoints": ["nose"], "space": ["x", "y"]},
    )
    confidence = xr.DataArray(
        np.ones((2, 1)),
        dims=("frame", "keypoints"),
        coords={"frame": [0, 1], "keypoints": ["nose"]},
    )

    aligned = align_pose_to_video(
        np.array([10.0, 11.0]),
        {"position": position, "confidence": confidence},
    )

    assert aligned["position"].dims[0] == "Time"
    np.testing.assert_array_equal(aligned["position"].coords["Time"], [10.0, 11.0])

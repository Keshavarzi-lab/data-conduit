# Align DeepLabCut pose to video time

DeepLabCut output is indexed by frame number; it does not carry the acquisition
clock used by the rest of a session. The DLC integration deliberately separates
reading from alignment:

- `DLCPose` / `read_dlc_pose` load frame-indexed pose arrays.
- `align_pose_to_video` attaches one camera timestamp to each frame.

This example builds the same array shapes in memory.

```python
import numpy as np
import xarray as xr

from data_conduit.integrations.DLC.pose import align_pose_to_video

position = xr.DataArray(
    np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[11.0, 20.5], [31.0, 40.5]],
            [[12.0, 21.0], [32.0, 41.0]],
            [[13.0, 21.5], [33.0, 41.5]],
        ]
    ),
    dims=("frame", "keypoints", "space"),
    coords={
        "keypoints": ["nose", "tail-base"],
        "space": ["x", "y"],
    },
    name="position",
)

confidence = xr.DataArray(
    np.array(
        [
            [0.99, 0.95],
            [0.98, 0.94],
            [0.97, 0.96],
            [0.99, 0.97],
        ]
    ),
    dims=("frame", "keypoints"),
    coords={"keypoints": ["nose", "tail-base"]},
    name="confidence",
)

video_times = np.array([100.00, 100.02, 100.04, 100.06])

aligned = align_pose_to_video(
    video_times,
    {"position": position, "confidence": confidence},
)

print(aligned["position"].dims)
print(aligned["position"].coords["Time"].values)
```

The `frame` dimension becomes `Time`, and the same timestamp coordinate is
applied to position and confidence. Alignment is positional: frame *i* receives
video timestamp *i*.

## Frame count is the alignment guarantee

Pose and video must contain exactly the same number of frames. A mismatch raises
instead of truncating or padding, because it may indicate a dropped video frame,
a stale DLC result, or the wrong pair of files.

If timestamps step backwards between video segments, the default behavior is to
warn and retain filename order so the clock reset remains visible. Use
`on_rollback="error"` when downstream processing requires monotonic time.

## Use alignment as a catalog configurator

Video and pose are independent readers. A configurator expresses their
dependency after both are available:

```python
from pathlib import Path

import pandas as pd

from data_conduit.datastructures import StreamCatalog
from data_conduit.integrations.DLC.pose import DLCPose, align_pose_to_video


def read_video_times(session_path: Path) -> pd.DataFrame:
    return pd.read_csv(session_path / "video-times.csv", index_col="Time")


catalog = StreamCatalog()
catalog.add_reader("video", read_video_times)
catalog.add_reader(
    "dlc",
    lambda session_path: DLCPose(
        experiment_directory_path=session_path,
        file_format="auto",
    ),
    optional=True,
)


def align_dlc(objects: dict[str, object]) -> dict[str, object]:
    configured = dict(objects)
    if "video" in configured and "dlc" in configured:
        configured["dlc"] = align_pose_to_video(
            configured["video"],
            configured["dlc"],
        )
    return configured


catalog.add_configurator("align_dlc", align_dlc)
```

The aligned mapping is normalised into `dlc:position` and `dlc:confidence`
streams. When a `DataStructure` combines sessions, the `Time` axis and session
provenance remain attached.

For interoperability with the `movement` ecosystem,
`pose_to_movement(position, confidence)` packages the aligned arrays into a
movement-schema xarray `Dataset` without changing their timestamps.

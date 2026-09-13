"""
Read a video frame for the spatial and trajectory figure backgrounds.
-------------------------------------------------------------------

Description:
    Read the first frame from an explicitly selected recording directory.
    Choose the video folder whose coordinates match the pose measurements.

Contents:
--------------------------------
- read_session_video_frame: Return the first RGB frame and its source path.
"""


################################################################################
# Imports
################################################################################

from pathlib import Path

import numpy as np

################################################################################


################################################################################
# Public API
################################################################################


# ===============================================================================
# 1| Read the Figure's Background Frame
# ===============================================================================


def read_session_video_frame(
    session_path: str | Path,                     # One recording's directory.
    *,
    video_subdir: str = "UndistortedVideoData",    # Matches the lab figure notebooks' DLC coordinates.
) -> tuple[np.ndarray, Path]:
    """
    Read the first frame without resizing or changing its coordinates.

    Parameters
    ----------
    session_path : str | Path
        Selected recording directory, available from DataStructure.sessions.
    video_subdir : str
        Video folder matching the tracking; default 'UndistortedVideoData'.

    Returns
    -------
    tuple[numpy.ndarray, pathlib.Path]
        RGB image with shape (height, width, 3), followed by its video path.

    Raises
    ------
    FileNotFoundError
        The chosen folder contains no supported video files.
    ValueError
        The decoded frame has an unsupported shape.
    """
    import imageio.v3 as iio  # Video decoding is needed only when reading a background.

    # === 1| Locate the First Video Chunk =========================================

    video_dir = Path(session_path) / video_subdir
    video_files = sorted(
        path for path in video_dir.glob("*")
        if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
    )  # Filename order matches the existing split-recording readers.
    if not video_files:
        raise FileNotFoundError(f"No video found in {video_dir}; choose the folder used for tracking.")

    # === 2| Decode Frame Zero and Return RGB =====================================

    video_path = video_files[0]
    frame = iio.imread(video_path, index=0)  # The figures use the first frame as an arena background.
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4) or min(frame.shape[:2]) == 0:
        raise ValueError(f"Video frame has an unsupported image shape: {frame.shape}.")
    return frame[..., :3], video_path  # Drop alpha when present; retain the original pixel geometry.


# ===============================================================================

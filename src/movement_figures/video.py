"""
Read a video frame for the spatial and trajectory figure backgrounds.
-------------------------------------------------------------------

Description:
    Read the first frame from an explicitly selected recording directory.
    Choose the video folder whose coordinates match the pose measurements.

Contents:
--------------------------------
- read_session_video_frame: Return the first RGB frame and its source path.
- resolve_static_roi: Choose a measured pixel location or the video's top centre.
"""


################################################################################
# Imports
################################################################################

from pathlib import Path
from collections.abc import Mapping
from numbers import Integral

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

# ===============================================================================
# 2 | Resolve a Static ROI in the First Frame's Pixel Coordinates
# ===============================================================================

def resolve_static_roi(
    background_frame: np.ndarray,
    *,
    roi_px: tuple[float, float] | np.ndarray | None = None,
    nosepoke: int | None = None,
    nosepoke_positions_px: Mapping[int, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, str]:
    """Return a static target position and a label describing its source.

    Parameters
    ----------
    background_frame : numpy.ndarray
        Nonempty RGB/RGBA frame with shape (height, width, channels). Its original
        pixel coordinates define x rightwards and y downwards; no resizing or
        arena geometry is inferred.
    roi_px : pair of float, numpy.ndarray, or None
        Optional measured (x, y) position in this frame. Supply either this point
        or ``nosepoke``, not both. None leaves the nosepoke/default choice active.
    nosepoke : int or None
        Port identifier to look up in ``nosepoke_positions_px``. IDs follow the
        supplied calibration's numbering; no zero/one-based conversion occurs.
        None with no measured point selects the video's top-centre pixel.
    nosepoke_positions_px : mapping of int to (float, float), or None
        Measured port locations for this recording and camera geometry. Port
        numbers alone do not establish locations. An explicitly selected port
        must be present in the mapping; the function never guesses its angle.

    Returns
    -------
    position_px : numpy.ndarray
        Independent float array of shape (2,) containing (x, y) in original
        pixels. The default is ((width - 1) / 2, 0), at the top of the video.
        Apply the recording's pixels-to-centimetres scale after this selection.
    label : str
        ``"North / top of video"``, ``"Measured static ROI"``, or
        ``"Nosepoke n"``. North is image-up, not an inferred compass direction or
        a claim about which physical arena port occupies that location.

    Raises
    ------
    ValueError
        Frame/point geometry is invalid, the point is outside the image, both
        selection modes are supplied, or the selected nosepoke lacks calibration.
    """
    # --- 2.1 | Validate the camera geometry without modifying the frame ---------
    frame = np.asarray(background_frame)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4) or min(frame.shape[:2]) < 1:
        raise ValueError("background_frame must be a nonempty RGB or RGBA image.")
    height, width = frame.shape[:2]
    if roi_px is not None and nosepoke is not None:
        raise ValueError("Choose roi_px or nosepoke, not both.")

    # --- 2.2 | Use measured port/point coordinates, otherwise image north -------
    if nosepoke is not None:
        if isinstance(nosepoke, (bool, np.bool_)) or not isinstance(nosepoke, Integral):
            raise ValueError("nosepoke must be an integer port identifier or None.")
        if nosepoke_positions_px is None or nosepoke not in nosepoke_positions_px:
            raise ValueError(
                f"Nosepoke {nosepoke} needs measured pixel coordinates in nosepoke_positions_px. "
                "Set nosepoke=None to use North / top of video."
            )
        point = nosepoke_positions_px[nosepoke]
        label = f"Nosepoke {nosepoke}"
    elif roi_px is not None:
        point, label = roi_px, "Measured static ROI"
    else:
        point, label = ((width - 1) / 2, 0.0), "North / top of video"

    # --- 2.3 | Reject invalid coordinates before any bearing is calculated ------
    position = np.array(point, dtype=float, copy=True)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("A static ROI must contain two finite pixel coordinates.")
    if not (0 <= position[0] <= width - 1 and 0 <= position[1] <= height - 1):
        raise ValueError("Static ROI coordinates must lie inside the video frame.")
    return position, label

'''
DeepLabCut pose loading, aligned to camera frame times.
-------------------------------------------------------

Description:
    DeepLabCut (DLC) writes one row of pose estimates per VIDEO FRAME, indexed
    by frame number (0, 1, 2, ...). It carries no wall-clock / experiment time
    of its own. The experiment clock for those frames lives elsewhere: the
    Bonsai ``VideoData`` CSV logs one row per frame too, and its ``Seconds``
    column is the HARP/Bonsai timestamp of each frame -- the SAME clock that
    ExperimentEvents, Nosepoke, SoundCard, etc. are on.

    So aligning DLC to the rest of a session is NOT a cross-clock TTL problem;
    it is a positional join: DLC row i takes the camera frame time of
    ``VideoData`` row i. Once that join is done the pose data sits on the shared
    session clock and flows through the ordinary sessiongroups pipeline
    (load_session -> [normalise_to_zero] -> combine_sessions) like any other
    same-clock structure.

    A session may split its recording into several file segments (e.g. Bonsai
    rolls the video / CSV partway through), producing several DLC outputs and
    several ``VideoData`` CSVs. Each DLC file is paired with the ``VideoData``
    file of matching length and the segments are concatenated in time order.
    The length match is also a sanity check: a DLC file whose row count does
    not match any VideoData file hints at a dropped-frame / wrong-file problem,
    so it is surfaced (raise or warn).

    Output shape (movement-friendly):
      * ``position``   : xr.DataArray (Time x keypoints x space[x, y])
      * ``confidence`` : xr.DataArray (Time x keypoints)
    These two arrays are the natural split used by the neuroinformatics
    ``movement`` package (position vs. likelihood), kept here on data-conduit's
    ``Time`` axis so they combine with the rest of a session. ``pose_to_movement``
    converts the pair into a ``movement``-schema ``xr.Dataset`` when needed.

Contents:
--------------------------------
- read_dlc_pose:    Load + frame-time-align one session's DLC pose.
- DLCPose:          Source-style wrapper exposing ``.data_arrays`` for the catalog.
- pose_to_movement: Convert (position, confidence) into a movement-schema Dataset.
'''





################################################################################
# Imports
################################################################################

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

################################################################################



# Constant: the three DLC per-keypoint sub-columns. ``x``/``y`` make up the
# spatial position; ``likelihood`` is the detection confidence kept separately.
_SPACE_COORDS = ('x', 'y')
_LIKELIHOOD_COORD = 'likelihood'




################################################################################
# Private Helpers
################################################################################



#===============================================================================
# 1| List the DLC Output Files in a Session
#===============================================================================
def _dlc_files(
        dlc_dir: Path,
        file_format: str,
) -> list[Path]:
    '''
    Return the DLC output files in ``dlc_dir`` for the requested format.

    Prefers ``.h5`` (faster, exact dtypes) and falls back to ``.csv`` when no
    HDF5 files are present, unless a format is pinned explicitly. DLC also
    writes ``*_meta.pickle`` / ``*_full.pickle`` sidecars; those are ignored.

    ----------
    Parameters:
        dlc_dir (Path):
            The session's ``DLC`` subdirectory.
        file_format (str):
            ``'h5'``, ``'csv'``, or ``'auto'`` (h5 if any, else csv).
    Returns:
        list[Path]:
            Matching DLC files, sorted by name.
    '''

    # Resolve 'auto' to whichever format actually has files present.
    if file_format == 'auto':
        file_format = 'h5' if list(dlc_dir.glob('*.h5')) else 'csv'

    if file_format not in ('h5', 'csv'):
        raise ValueError(f"file_format must be 'h5', 'csv', or 'auto' (got {file_format!r}).")

    return sorted(dlc_dir.glob(f'*.{file_format}'))

#===============================================================================



#===============================================================================
# 2| Read One DLC Table (h5 or csv) Into a 2-Level-Column DataFrame
#===============================================================================
def _read_dlc_table(
        path: Path,
) -> pd.DataFrame:
    '''
    Read one DLC file into a DataFrame with (bodyparts, coords) columns.

    DLC files carry a 3-level column header (scorer, bodyparts, coords) and a
    frame-number index. The single-valued ``scorer`` level is dropped so the
    columns are just ``(bodypart, {x, y, likelihood})``.

    ----------
    Parameters:
        path (Path):
            A DLC ``.h5`` or ``.csv`` output file.
    Returns:
        pd.DataFrame:
            One row per frame; columns are a 2-level MultiIndex
            ``(bodyparts, coords)``.
    '''

    # h5 stores the frame index + 3-level columns natively; the csv carries 3
    # header rows (scorer / bodyparts / coords) with the frame index in column 0.
    df = pd.read_hdf(path) if path.suffix == '.h5' else pd.read_csv(path, header=[0, 1, 2], index_col=0)

    # Drop the scorer level if present (single model -> single scorer).
    if df.columns.nlevels == 3:
        df.columns = df.columns.droplevel(0)
    df.columns = df.columns.set_names(['bodyparts', 'coords'])
    return df

#===============================================================================



#===============================================================================
# 3| Read Per-Segment Camera Frame Times From the VideoData CSVs
#===============================================================================
def _video_frame_times(
        video_dir: Path,
        time_column: str,
) -> list[tuple[Path, np.ndarray]]:
    '''
    Return ``(path, frame_times)`` for each VideoData CSV in ``video_dir``.

    ``frame_times`` is the CSV's ``time_column`` (default ``'Seconds'``): the
    HARP/Bonsai timestamp of every camera frame, one value per row. These are
    the times the DLC rows are aligned to.

    ----------
    Parameters:
        video_dir (Path):
            The session's ``VideoData`` subdirectory.
        time_column (str):
            Column holding the per-frame timestamp. Default ``'Seconds'``.
    Returns:
        list[tuple[Path, np.ndarray]]:
            One ``(csv_path, times)`` pair per VideoData CSV.
    '''

    segments = []
    for csv_path in sorted(video_dir.glob('*.csv')):
        times = pd.read_csv(csv_path, usecols=[time_column])[time_column].to_numpy(dtype=float)
        segments.append((csv_path, times))
    return segments

#===============================================================================



#===============================================================================
# 4| Pair Each DLC File With the VideoData Segment of Matching Length
#===============================================================================
def _pair_by_length(
        dlc_paths: list[Path],
        video_segments: list[tuple[Path, np.ndarray]],
        *,
        on_length_mismatch: str,
) -> list[tuple[pd.DataFrame, np.ndarray]]:
    '''
    Match each DLC table to the VideoData segment with the same frame count.

    Each ``VideoData`` segment is consumed at most once. A DLC file whose row
    count matches no remaining segment is a discrepancy (likely a dropped
    frame, a stale DLC re-run, or a mis-paired file); ``on_length_mismatch``
    decides whether that raises or merely warns (and skips the file).

    ----------
    Parameters:
        dlc_paths (list[Path]):
            The session's DLC output files.
        video_segments (list[tuple[Path, np.ndarray]]):
            ``(path, frame_times)`` pairs from ``_video_frame_times``.
        on_length_mismatch (str):
            ``'error'`` to raise, ``'warn'`` to warn and skip the unmatched
            DLC file.
    Returns:
        list[tuple[pd.DataFrame, np.ndarray]]:
            ``(dlc_table, frame_times)`` pairs, one per matched DLC file,
            ordered by each segment's first frame time.
    '''

    # Index the still-available video segments by their frame count. A list per
    # length handles the (rare) case of two segments sharing a length.
    available: dict[int, list[tuple[Path, np.ndarray]]] = {}
    for path, times in video_segments:
        available.setdefault(len(times), []).append((path, times))

    pairs: list[tuple[pd.DataFrame, np.ndarray]] = []
    for dlc_path in dlc_paths:
        table = _read_dlc_table(dlc_path)
        n = len(table)

        # No remaining VideoData segment of this length -> discrepancy.
        if n not in available or not available[n]:
            video_lengths = sorted(len(t) for _, t in video_segments)
            message = (
                f'DLC file {dlc_path.name!r} has {n} frames, which matches no '
                f'remaining VideoData segment (segment frame counts: {video_lengths}). '
                f'This usually means a dropped-frame mismatch or a wrong/stale DLC file.'
            )
            if on_length_mismatch == 'error':
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
            continue

        _, times = available[n].pop(0)
        pairs.append((table, times))

    if not pairs:
        raise ValueError('no DLC file could be paired with a VideoData segment of matching length.')

    # Concatenate segments in chronological (camera-time) order.
    pairs.sort(key=lambda pair: float(pair[1][0]) if pair[1].size else np.inf)
    return pairs

#===============================================================================



#===============================================================================
# 5| Build Per-Segment position / confidence DataArrays
#===============================================================================
def _segment_to_arrays(
        table: pd.DataFrame,
        frame_times: np.ndarray,
        time_coord: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    '''
    Turn one (DLC table, frame_times) pair into position + confidence arrays.

    ----------
    Parameters:
        table (pd.DataFrame):
            One segment's DLC table with ``(bodyparts, coords)`` columns.
        frame_times (np.ndarray):
            The camera frame times for this segment (becomes the time axis).
        time_coord (str):
            Name to give the time dimension (default caller-supplied ``'Time'``).
    Returns:
        tuple[xr.DataArray, xr.DataArray]:
            ``(position, confidence)`` for this segment.
    '''

    # Split out x / y / likelihood; each is (frames x bodyparts). Reindex y and
    # likelihood to x's bodypart order so the stack lines up regardless of the
    # column order DLC happened to write.
    xs = table.xs('x', level='coords', axis=1)
    keypoints = list(xs.columns)
    ys = table.xs('y', level='coords', axis=1)[keypoints]
    likelihood = table.xs(_LIKELIHOOD_COORD, level='coords', axis=1)[keypoints]

    # position: stack x and y along a new 'space' axis -> (frames, keypoints, 2).
    position_values = np.stack([xs.to_numpy(dtype=float), ys.to_numpy(dtype=float)], axis=-1)
    position = xr.DataArray(
        position_values,
        dims=(time_coord, 'keypoints', 'space'),
        coords={time_coord: frame_times, 'keypoints': keypoints, 'space': list(_SPACE_COORDS)},
        name='position',
    )

    confidence = xr.DataArray(
        likelihood.to_numpy(dtype=float),
        dims=(time_coord, 'keypoints'),
        coords={time_coord: frame_times, 'keypoints': keypoints},
        name='confidence',
    )
    return position, confidence

#===============================================================================



################################################################################




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| read_dlc_pose (Load + Frame-Time-Align One Session's Pose)
#===============================================================================
def read_dlc_pose(
        experiment_directory_path: str | Path,
        *,
        dlc_subdir: str = 'DLC',
        video_subdir: str = 'VideoData',
        time_column: str = 'Seconds',
        time_coord: str = 'Time',
        file_format: str = 'auto',
        on_length_mismatch: str = 'error',
) -> dict[str, xr.DataArray]:
    '''
    Load one session's DLC pose, aligned to camera frame times.

    Reads every DLC output in ``<session>/<dlc_subdir>``, pairs each with the
    ``<session>/<video_subdir>`` CSV of matching frame count, attaches that
    CSV's per-frame timestamps as the time axis, and concatenates the segments
    in time order. The result sits on the shared session clock (the same clock
    as ExperimentEvents), so it merges with the rest of the session directly.

    ----------
    Parameters:
        experiment_directory_path (str | Path):
            One session directory (the folder that contains ``DLC`` and
            ``VideoData`` subfolders).
        dlc_subdir (str):
            Name of the DLC output subfolder. Default ``'DLC'``.
        video_subdir (str):
            Name of the camera-metadata subfolder. Default ``'VideoData'``.
        time_column (str):
            Column in the VideoData CSV holding each frame's timestamp.
            Default ``'Seconds'``.
        time_coord (str):
            Name to give the time dimension. Default ``'Time'`` (matches the
            sessiongroups alignment pipeline).
        file_format (str):
            ``'h5'``, ``'csv'``, or ``'auto'`` (h5 if present, else csv).
        on_length_mismatch (str):
            ``'error'`` (default) to raise when a DLC file matches no
            VideoData segment by length; ``'warn'`` to warn and skip it.
    Returns:
        dict[str, xr.DataArray]:
            ``{'position': (Time x keypoints x space), 'confidence':
            (Time x keypoints)}``.
    '''

    session = Path(experiment_directory_path)
    dlc_dir = session / dlc_subdir
    video_dir = session / video_subdir

    if not dlc_dir.is_dir():
        raise FileNotFoundError(f'no {dlc_subdir!r} folder in session {session}.')
    if not video_dir.is_dir():
        raise FileNotFoundError(f'no {video_subdir!r} folder in session {session}.')

    dlc_paths = _dlc_files(dlc_dir, file_format)
    if not dlc_paths:
        raise FileNotFoundError(f'no DLC {file_format!r} files in {dlc_dir}.')

    video_segments = _video_frame_times(video_dir, time_column)
    if not video_segments:
        raise FileNotFoundError(f'no VideoData CSVs in {video_dir}.')

    # Pair, build per-segment arrays, then concatenate along the time axis.
    pairs = _pair_by_length(dlc_paths, video_segments, on_length_mismatch=on_length_mismatch)
    positions, confidences = [], []
    for table, frame_times in pairs:
        position, confidence = _segment_to_arrays(table, frame_times, time_coord)
        positions.append(position)
        confidences.append(confidence)

    position = positions[0] if len(positions) == 1 else xr.concat(positions, dim=time_coord)
    confidence = confidences[0] if len(confidences) == 1 else xr.concat(confidences, dim=time_coord)
    return {'position': position, 'confidence': confidence}

#===============================================================================



#===============================================================================
# 2| DLCPose (Source-Style Wrapper for the DataStructureCatalog)
#===============================================================================
class DLCPose:
    '''
    Thin source-style wrapper around ``read_dlc_pose``.

    Exposes the loaded pose as a ``.data_arrays`` mapping, the same interface
    the HARP presets (Nosepoke, SoundCard, ...) expose. That means it drops
    straight into a ``DataStructureCatalog`` spec::

        catalog.add(DataStructureSpec(
            name='dlc',
            reader=lambda p: DLCPose(experiment_directory_path=p),
        ))

    and the loader turns its two arrays into the bundle members
    ``'dlc:position'`` and ``'dlc:confidence'``. Because the data already
    carries camera-frame times on the session clock, ``sync`` stays ``None``
    (it is a same-clock structure -- no TTL conversion needed).

    ----------
    Parameters:
        experiment_directory_path (str | Path):
            One session directory. Required.
        **kwargs:
            Forwarded to ``read_dlc_pose`` (``dlc_subdir``, ``video_subdir``,
            ``time_column``, ``time_coord``, ``file_format``,
            ``on_length_mismatch``).

    Attributes:
        data_arrays (dict[str, xr.DataArray]):
            ``{'position': ..., 'confidence': ...}``.
    '''

    def __init__(
            self,
            experiment_directory_path: str | Path | None = None,
            **kwargs,
    ) -> None:
        '''Load the session's pose into ``self.data_arrays``.'''
        if experiment_directory_path is None:
            raise ValueError('experiment_directory_path is required.')
        self.experiment_directory_path = Path(experiment_directory_path)
        self.data_arrays = read_dlc_pose(self.experiment_directory_path, **kwargs)

#===============================================================================



#===============================================================================
# 3| pose_to_movement (Convert to a movement-Schema Dataset)
#===============================================================================
def pose_to_movement(
        position: xr.DataArray,
        confidence: xr.DataArray,
        *,
        individual: str = 'individual_0',
        time_coord: str = 'Time',
        source_software: str = 'DeepLabCut',
        fps: float | None = None,
):
    '''
    Pack ``position`` + ``confidence`` into a ``movement``-schema ``xr.Dataset``.

    The neuroinformatics ``movement`` package expects a poses dataset with
    dims ``(time, individuals, keypoints, space)`` and data variables
    ``position`` and ``confidence``. data-conduit keeps the arrays on a
    ``Time`` axis (and possibly with extra coords such as the session
    ``label`` after a combine); this renames the time dim, adds the singleton
    ``individuals`` dimension, orders the dims, and sets the attributes
    ``movement`` looks for. Any extra coordinates riding the time axis (e.g.
    ``label``) are preserved.

    ----------
    Parameters:
        position (xr.DataArray):
            ``(Time x keypoints x space)`` positions.
        confidence (xr.DataArray):
            ``(Time x keypoints)`` likelihoods.
        individual (str):
            Name for the single tracked individual. Default ``'individual_0'``.
        time_coord (str):
            Current name of the time dimension to rename to ``'time'``.
            Default ``'Time'``.
        source_software (str):
            Value for the dataset's ``source_software`` attribute.
        fps (float | None):
            Optional frames-per-second to record in ``attrs``.
    Returns:
        xr.Dataset:
            A ``movement``-compatible poses dataset.
    '''

    rename = {time_coord: 'time'} if time_coord in position.dims else {}
    position = position.rename(rename).expand_dims({'individuals': [individual]})
    confidence = confidence.rename(rename).expand_dims({'individuals': [individual]})

    position = position.transpose('time', 'individuals', 'keypoints', 'space', ...)
    confidence = confidence.transpose('time', 'individuals', 'keypoints', ...)

    dataset = xr.Dataset({'position': position, 'confidence': confidence})
    dataset.attrs['source_software'] = source_software
    if fps is not None:
        dataset.attrs['fps'] = fps
    return dataset

#===============================================================================



################################################################################

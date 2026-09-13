'''
DeepLabCut pose loading (frame-indexed) and alignment to camera frame times.
----------------------------------------------------------------------------



Description:
    DeepLabCut (DLC) writes one row of pose estimates per VIDEO FRAME, indexed
    by frame number (0, 1, 2, ...). It carries no wall-clock / experiment time
    of its own. The experiment clock for those frames lives elsewhere: the
    Bonsai ``VideoData`` CSV logs one row per frame too, and its ``Seconds``
    column is the HARP/Bonsai timestamp of each frame (the SAME clock that
    ExperimentEvents, Nosepoke, SoundCard, etc. are on).

    This module keeps those two jobs separate, by design:

      * READING (a datasource).  ``DLCPose`` / ``read_dlc_pose`` read a session's
        DLC output(s) into FRAME-INDEXED arrays. They know nothing about video or
        time. A session split across several DLC files is concatenated in FILENAME
        order and given one continuous ``0..N-1`` frame index (so the assembled
        pose is monotonic, not a per-file reset). This is an ordinary source
        object exposing ``.data_arrays``, just like the HARP presets.

      * ALIGNING (a datastructure step).  ``align_pose_to_video`` takes a loaded
        ``VideoData`` (or its flattened table) and a loaded ``DLCPose`` (or its
        arrays), checks the two carry the SAME number of frames, and stamps the
        video's per-frame ``Seconds`` onto the pose by position, returning
        TIME-INDEXED arrays on the shared session clock. This requires the DLC
        and video chunks to represent the same frames in the same order. Equal
        total counts detect a length mismatch; they cannot prove that the
        selected files are the matching recordings.

    Splitting the two responsibilities means there is ONE read of the VideoData
    per session (the alignment reuses the already-loaded video rather than
    re-reading the CSVs), the segment ordering is decided ONCE (filename order,
    on both sides), and the frame -> time join lives at the datastructure level
    where both streams are visible, instead of being buried inside the pose
    reader.

    Output shape (movement-friendly):
      * ``position``   : xr.DataArray (frame|Time x keypoints x space[x, y])
      * ``confidence`` : xr.DataArray (frame|Time x keypoints)
    ``read_dlc_pose`` returns these on a ``frame`` axis; ``align_pose_to_video``
    returns them on a ``Time`` axis. These two arrays are the natural split used
    by the neuroinformatics ``movement`` package (position vs. likelihood);
    ``pose_to_movement`` converts the pair into a ``movement``-schema
    ``xr.Dataset`` when needed.

Contents:
--------------------------------
- read_dlc_pose:       Load one session's DLC pose as frame-indexed arrays.
- DLCPose:             Source-style wrapper exposing ``.data_arrays`` for the catalog.
- align_pose_to_video: Stamp VideoData frame times onto frame-indexed pose (with checks).
- pose_to_movement:    Convert (position, confidence) into a movement-schema Dataset.
'''





################################################################################
# Imports
################################################################################

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data_conduit.core.utils.utils_core import _concat_split_dataframes

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

    Prefers ``.csv`` and falls back to ``.h5`` when no CSV files are present,
    unless a format is pinned explicitly. DLC also
    writes ``*_meta.pickle`` / ``*_full.pickle`` sidecars; those are ignored.

    ----------
    Parameters:
        dlc_dir (Path):
            The session's ``DLC`` subdirectory.
        file_format (str):
            ``'csv'``, ``'h5'``, or ``'auto'`` (csv if any, else h5).
    Returns:
        list[Path]:
            Matching DLC files, sorted by name (so a multi-file session is read
            in filename, hence chronological, order).
    '''

    # Resolve 'auto' to whichever format actually has files present, preferring
    # CSV because that is the lab's canonical DLC export for this workflow.
    if file_format == 'auto':
        file_format = 'csv' if list(dlc_dir.glob('*.csv')) else 'h5'

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
# 3| Turn One DLC Table Into Frame-Indexed position / confidence Arrays
#===============================================================================
def _table_to_arrays(
        table: pd.DataFrame,
        frame_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    '''
    Turn one DLC table into frame-indexed ``position`` + ``confidence`` arrays.

    No times are attached here. The arrays carry a bare ``frame_dim`` axis with
    NO coordinate; a continuous frame index is assigned once in
    ``read_dlc_pose`` after all of a session's files are concatenated.

    ----------
    Parameters:
        table (pd.DataFrame):
            One DLC file's table with ``(bodyparts, coords)`` columns.
        frame_dim (str):
            Name to give the per-frame dimension (e.g. ``'frame'``).
    Returns:
        tuple[xr.DataArray, xr.DataArray]:
            ``(position, confidence)`` for this file, where ``position`` is
            ``(frame x keypoints x space)`` and ``confidence`` is
            ``(frame x keypoints)``.
    '''

    # Split out x / y / likelihood; each is (frames x bodyparts). Reindex y and
    # likelihood to x's bodypart order so the stack lines up regardless of the
    # column order DLC happened to write.
    xs = table.xs('x', level='coords', axis=1)
    keypoints = list(xs.columns)
    ys = table.xs('y', level='coords', axis=1)[keypoints]
    likelihood = table.xs(_LIKELIHOOD_COORD, level='coords', axis=1)[keypoints]

    # position: stack x and y along a new 'space' axis -> (frames, keypoints, 2).
    # We deliberately leave frame_dim coordinate-free; it is filled after concat.
    position_values = np.stack([xs.to_numpy(dtype=float), ys.to_numpy(dtype=float)], axis=-1)
    position = xr.DataArray(
        position_values,
        dims=(frame_dim, 'keypoints', 'space'),
        coords={'keypoints': keypoints, 'space': list(_SPACE_COORDS)},
        name='position',
    )

    confidence = xr.DataArray(
        likelihood.to_numpy(dtype=float),
        dims=(frame_dim, 'keypoints'),
        coords={'keypoints': keypoints},
        name='confidence',
    )
    return position, confidence

#===============================================================================



#===============================================================================
# 4| Extract Per-Frame Video Times From a VideoData Object / Table
#===============================================================================
def _video_times(
        video,
) -> np.ndarray:
    '''
    Pull the per-frame camera times out of a VideoData object (or table).

    Accepts whatever the caller has to hand: a ``VideoData`` source object (its
    ``.df`` is used), a pre-flattened DataFrame (its ``Time`` index is the
    per-frame clock, because VideoData is read with the ``Seconds`` column as
    ``index_col=0`` and renamed to ``Time``), a Series, or a bare array of
    times. The result is the one-value-per-frame timestamp vector the pose is
    aligned to.

    Only a raw ``VideoData`` object is flattened here (its ``.df`` may be a
    multi-file dict), using filename order and retaining each file's acquired row
    order. Anything already flattened (e.g. a loaded ``video`` bundle member) also
    keeps the order given. Sorting these times without reordering the matching
    pose frames would pair different samples and could conceal a clock reset.
    The values must be real and finite; ``align_pose_to_video`` applies its
    existing ``on_rollback`` policy to any decrease in the preserved clock.

    ----------
    Parameters:
        video (object | pandas.DataFrame | pandas.Series | array-like):
            A ``VideoData`` source object, its flattened table, or a times array.
    Returns:
        numpy.ndarray:
            1D float array of per-frame times, one per camera frame.
    Raises:
        TypeError, ValueError:
            If camera times are not a one-dimensional, finite real numeric array.
    '''

    # A VideoData source object carries the table on ``.df`` and may be split
    # across files (a dict), so flatten it with the shared filename-first policy.
    # on_rollback='ignore' because align_pose_to_video does its own (configurable)
    # rollback check on the final Time axis, so we do not want a second warning here.
    # Anything else is assumed to be ALREADY flattened, so we trust its order.
    if hasattr(video, 'df'):
        table = _concat_split_dataframes(
            video.df,
            on_rollback='ignore',                   # The aligner retains control of its warn/error/ignore policy.
            sort_by_time=False,                     # Camera rows must remain paired with the unchanged pose frame order.
        )
    else:
        table = video

    # The per-frame clock is the Time index for a DataFrame, the values for a
    # Series, or the array itself otherwise.
    if isinstance(table, pd.DataFrame):
        times = table.index.to_numpy()
    elif isinstance(table, pd.Series):
        times = table.to_numpy()
    else:
        times = np.asarray(table)

    # Reject complex values before conversion to float could discard their
    # imaginary part. Calendar/timedelta and text values are not acquired seconds.
    if not (np.issubdtype(times.dtype, np.integer) or np.issubdtype(times.dtype, np.floating)):
        raise TypeError('Video times must contain real numeric acquired seconds.')
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError('Video times must be a one-dimensional array of finite acquired seconds.')

    return times.astype(float, copy=False)           # Preserve the existing float output without changing row order.

#===============================================================================



################################################################################




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| read_dlc_pose (Load One Session's Pose as Frame-Indexed Arrays)
#===============================================================================
def read_dlc_pose(
        experiment_directory_path: str | Path,
        *,
        dlc_subdir: str = 'DLC',
        file_format: str = 'csv',
        frame_dim: str = 'frame',
) -> dict[str, xr.DataArray]:
    '''
    Load one session's DLC pose into frame-indexed arrays (no times attached).

    Reads every DLC output in ``<session>/<dlc_subdir>`` in filename order,
    concatenates them, and gives the whole session one continuous ``0..N-1``
    frame index. This is a pure reader: it does not look at VideoData and does
    not put pose on the session clock. Use ``align_pose_to_video`` for that.

    ----------
    Parameters:
        experiment_directory_path (str | Path):
            One session directory (the folder that contains the ``DLC``
            subfolder).
        dlc_subdir (str):
            Name of the DLC output subfolder. Default ``'DLC'``.
        file_format (str):
            ``'csv'`` (default), ``'h5'``, or ``'auto'`` (csv if present, else h5).
        frame_dim (str):
            Name to give the per-frame dimension. Default ``'frame'``.
    Returns:
        dict[str, xr.DataArray]:
            ``{'position': (frame x keypoints x space), 'confidence':
            (frame x keypoints)}``, with ``frame_dim`` a continuous ``0..N-1``
            index spanning all of the session's DLC files.
    '''

    session = Path(experiment_directory_path)
    dlc_dir = session / dlc_subdir

    if not dlc_dir.is_dir():
        raise FileNotFoundError(f'no {dlc_subdir!r} folder in session {session}.')

    dlc_paths = _dlc_files(dlc_dir, file_format)
    if not dlc_paths:
        raise FileNotFoundError(f'no DLC {file_format!r} files in {dlc_dir}.')

    # 1| Read each DLC file (filename order) into frame-indexed arrays.
    positions, confidences = [], []
    for path in dlc_paths:
        table = _read_dlc_table(path)
        position, confidence = _table_to_arrays(table, frame_dim)
        positions.append(position)
        confidences.append(confidence)

    # 2| Concatenate the files along the frame axis (a no-op for a single file).
    position = positions[0] if len(positions) == 1 else xr.concat(positions, dim=frame_dim)
    confidence = confidences[0] if len(confidences) == 1 else xr.concat(confidences, dim=frame_dim)

    # 3| Give the assembled pose one continuous 0..N-1 frame index, so it is
    #    monotonic rather than restarting at 0 per file. This is the index the
    #    aligner replaces with the matching video Times.
    frame_index = np.arange(position.sizes[frame_dim])
    position = position.assign_coords({frame_dim: frame_index})
    confidence = confidence.assign_coords({frame_dim: frame_index})

    return {'position': position, 'confidence': confidence}

#===============================================================================



#===============================================================================
# 2| DLCPose (Source-Style Wrapper for the DataStructureCatalog)
#===============================================================================
class DLCPose:
    '''
    Thin source-style wrapper around ``read_dlc_pose``.

    Exposes the loaded pose as a ``.data_arrays`` mapping, the same interface
    the HARP presets (Nosepoke, SoundCard, ...) expose, so it drops straight
    into a ``DataStructureCatalog`` spec::

        catalog.add(DataStructureSpec(
            name='dlc',
            reader=lambda p: DLCPose(experiment_directory_path=p),
        ))

    and the loader turns its two arrays into the bundle members
    ``'dlc:position'`` and ``'dlc:confidence'``. NOTE that those members are
    FRAME-INDEXED: this wrapper does not attach camera times. To put pose on the
    session clock, pass this object (or its arrays) together with the session's
    ``VideoData`` to ``align_pose_to_video``.

    ----------
    Parameters:
        experiment_directory_path (str | Path):
            One session directory. Required.
        **kwargs:
            Forwarded to ``read_dlc_pose`` (``dlc_subdir``, ``file_format``,
            ``frame_dim``).

    Attributes:
        experiment_directory_path (Path):
            The session directory this pose was read from.
        data_arrays (dict[str, xr.DataArray]):
            ``{'position': ..., 'confidence': ...}``, frame-indexed.
    '''

    def __init__(
            self,
            experiment_directory_path: str | Path | None = None,
            **kwargs,
    ) -> None:
        '''
        Load the session's frame-indexed pose into ``self.data_arrays``.

        ----------
        Parameters:
            experiment_directory_path (str | Path | None):
                One session directory. Required (None raises).
            **kwargs:
                Forwarded to ``read_dlc_pose``.
        Returns:
            None.
        '''
        if experiment_directory_path is None:
            raise ValueError('experiment_directory_path is required.')
        self.experiment_directory_path = Path(experiment_directory_path)
        self.data_arrays = read_dlc_pose(self.experiment_directory_path, **kwargs)

#===============================================================================



#===============================================================================
# 3| align_pose_to_video (Stamp VideoData Times Onto Frame-Indexed Pose)
#===============================================================================
def align_pose_to_video(
        video,
        pose,
        *,
        frame_dim: str = 'frame',
        time_coord: str = 'Time',
        on_rollback: str = 'warn',
) -> dict[str, xr.DataArray]:
    '''
    Put frame-indexed pose on the session clock using the VideoData frame times.

    This is the datastructure-level join that the datasource deliberately does
    NOT do. It takes a session's loaded ``VideoData`` and ``DLCPose`` (the two
    streams of camera frames), checks they have the SAME number of frames, and
    stamps the video's per-frame time onto the pose by position. This assumes
    the filename-ordered pose and video chunks describe the same frames. Equal
    counts are required but do not prove that the selected files correspond.

    ----------
    Parameters:
        video (object | pandas.DataFrame | pandas.Series | array-like):
            The session's camera frame times. Either a ``VideoData`` source
            object (its ``.df`` is used), its flattened ``Time``-indexed table,
            or a bare array of per-frame times. Whatever is given is flattened
            in filename order and read as one time-per-frame vector.
        pose (object | dict[str, xr.DataArray]):
            The session's frame-indexed pose. Either a ``DLCPose`` object (its
            ``.data_arrays`` is used) or a ``{'position': ..., 'confidence':
            ...}`` mapping, both on a ``frame_dim`` axis.
        frame_dim (str):
            Name of the pose's per-frame dimension to convert. Default
            ``'frame'`` (matches ``read_dlc_pose``).
        time_coord (str):
            Name to give the resulting time dimension. Default ``'Time'``
            (matches the sessiongroups alignment pipeline).
        on_rollback (str):
            What to do when the resulting time axis is not increasing (the
            camera clock decreases within or between video files): ``'warn'`` (default) to
            warn and keep the data in filename order, ``'error'`` to raise, or
            ``'ignore'`` to do neither. This compatibility policy concerns
            decreases; equal timestamps are retained by this aligner. The Q_C
            catalog and movement conversion separately require strictly
            increasing times before their temporal analysis workflow.
    Returns:
        dict[str, xr.DataArray]:
            ``{'position': (Time x keypoints x space), 'confidence':
            (Time x keypoints)}`` on the shared session clock.
    '''

    # 1| Per-frame video times (flattened, filename order, one value per frame).
    times = _video_times(video)

    # 2| Frame-indexed pose arrays (accept a DLCPose object or a bare dict).
    arrays = pose.data_arrays if hasattr(pose, 'data_arrays') else pose
    position, confidence = arrays['position'], arrays['confidence']

    # 3| Reject different frame counts before joining by position. Matching
    #    totals alone do not establish file correspondence; the selected DLC
    #    and video chunks must also refer to the same recording frames.
    n_pose = position.sizes[frame_dim]
    if n_pose != len(times):
        raise ValueError(
            f'cannot align pose to video: pose has {n_pose} frames but VideoData has '
            f'{len(times)} frame times. DLC is run per video file, so the two must '
            f'correspond one-to-one; a mismatch points to a dropped frame, a '
            f'wrong/stale DLC file, or a missing VideoData segment.'
        )

    # 4| Stamp the times onto the pose by POSITION: rename the frame axis to the
    #    time axis and attach the per-frame times as its coordinate. Both arrays
    #    share the frame axis, so the same times land on both.
    position = position.rename({frame_dim: time_coord}).assign_coords({time_coord: times})
    confidence = confidence.rename({frame_dim: time_coord}).assign_coords({time_coord: times})

    # 5| Surface any decrease in the acquired camera clock, within or between
    #    files. Rows still follow filename/frame order so the reset remains
    #    visible and every timestamp stays attached to its original pose frame.
    #    Keep the existing direct-call warn/error/ignore policy.
    if on_rollback != 'ignore' and np.any(times[1:] < times[:-1]):
        message = (
            'aligned pose Time axis steps backwards; the camera clock appears to '
            'have reset within or between VideoData files. Pose is left in filename order '
            '(not time-sorted) so the reset stays visible; downstream steps '
            'assuming monotonic time may need attention.'
        )
        if on_rollback == 'error':
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)

    return {'position': position, 'confidence': confidence}

#===============================================================================



#===============================================================================
# 4| pose_to_movement (Convert to a movement-Schema Dataset)
#===============================================================================
def pose_to_movement(
        position: xr.DataArray,                         # One recording's aligned DLC x/y positions on its acquired Time axis.
        confidence: xr.DataArray,                       # Matching likelihoods for the same frames and keypoints.
        *,
        individual: str = 'individual_0',                # Label to add for the single animal tracked by this DLC export.
        time_coord: str = 'Time',                        # Input dimension containing acquired times in seconds.
        source_software: str = 'DeepLabCut',              # Tracking provenance recorded on the output dataset.
        fps: float | None = None,                        # Optional metadata; never used to replace the acquired timestamps.
) -> xr.Dataset:                                        # Returns an independent movement pose dataset for one recording.
    '''
    Pack aligned DLC position and confidence into a movement pose dataset.

    data-conduit stores separate arrays using ``Time`` and ``keypoints``.
    movement uses singular ``time``, ``keypoint`` and ``individual`` names.
    This function validates the paired arrays, renames those dimensions, and
    adds the one animal dimension. It preserves the acquired seconds and source
    coordinates; it does not reconstruct time from frame number or FPS.

    Convert one recording at a time. Different recordings can reuse the same
    clock values, so joining them before temporal analysis would mix unrelated
    samples. Raw data-conduit arrays remain unchanged, and the returned dataset
    has its own data buffers for subsequent optional filtering.

    Parameters
    ----------
    position : xr.DataArray
        DLC positions with dimensions ``(Time, keypoints, space)`` in any order.
        ``space`` must contain ``x`` and ``y`` in that order. NaNs represent
        missing tracking; infinite coordinates are rejected.
    confidence : xr.DataArray
        Likelihoods with dimensions ``(Time, keypoints)`` in any order. Shared
        coordinates must match position exactly. Values must be in [0, 1] or
        NaN; differing frame indexes must not be silently aligned by xarray.
    individual : str
        Non-empty label for the single animal in this DLC export. Default
        ``'individual_0'``; this function does not infer animal identity.
    time_coord : str
        Input time dimension. Default ``'Time'``. Its values must be finite,
        numeric seconds that increase strictly within this recording.
    source_software : str
        Tracking software name stored in the dataset attributes. Default
        ``'DeepLabCut'`` because these arrays originate from the DLC reader.
    fps : float | None
        Optional positive frame-rate metadata. None leaves FPS unspecified;
        the existing acquired time values are used in either case.

    Returns
    -------
    xr.Dataset
        ``position`` has dimensions ``(time, space, keypoint, individual)``;
        ``confidence`` has dimensions ``(time, keypoint, individual)``. Extra
        source coordinates are retained. Spatial units are pixels and time
        units are seconds. No filtering or lens correction is performed.
    '''

    # === 1| Check That the Two Arrays Describe the Same Frames and Keypoints =====

    if not isinstance(position, xr.DataArray) or not isinstance(confidence, xr.DataArray):
        raise TypeError('position and confidence must both be xarray DataArrays.')

    expected_position = (time_coord, 'keypoints', 'space')
    expected_confidence = (time_coord, 'keypoints')
    if sorted(position.dims) != sorted(expected_position):
        raise ValueError(f'position dimensions must be {expected_position}, in any order.')
    if sorted(confidence.dims) != sorted(expected_confidence):
        raise ValueError(f'confidence dimensions must be {expected_confidence}, in any order.')

    for name in confidence.coords:
        if name not in position.coords or not confidence[name].equals(position[name]):
            raise ValueError(f'position and confidence disagree on coordinate {name!r}.')

    for name in position.coords:
        if name != 'space' and name not in confidence.coords:
            raise ValueError(f'confidence is missing position coordinate {name!r}.')

    xr.align(position, confidence, join='exact', copy=False)  # Reject mismatched indexes before Dataset could join them automatically.

    # === 2| Check the Recording Clock, Animal Label, and Tracking Values =========

    if not isinstance(individual, str) or not individual:
        raise ValueError('individual must be a non-empty animal label.')
    if list(position.space.values) != ['x', 'y']:
        raise ValueError('position.space must contain ["x", "y"] in that order.')

    for name in (time_coord, 'keypoints'):
        if position.sizes[name] == 0 or not position.get_index(name).is_unique:
            raise ValueError(f'{name} must contain non-empty, unique coordinates.')

    time_dtype = position[time_coord].dtype
    if not (np.issubdtype(time_dtype, np.integer) or np.issubdtype(time_dtype, np.floating)):
        raise TypeError(f'{time_coord} must contain real numeric acquired seconds.')
    times = position[time_coord].to_numpy()                   # Read the existing clock without constructing a new frame/FPS time axis.
    # Direct comparison also detects decreases in unsigned integers, whose
    # subtraction would wrap to a large positive number instead of a negative gap.
    if not np.isfinite(times).all() or np.any(times[1:] <= times[:-1]):
        raise ValueError('Acquired times must be finite and strictly increasing within one recording.')

    if 'session' in position.coords:
        session_ids = np.asarray(position.session.values).reshape(-1)
        if len(np.unique(session_ids)) != 1:
            raise ValueError('Select one recording before converting its pose to movement.')

    for name, array in (('position', position), ('confidence', confidence)):
        if not np.issubdtype(array.dtype, np.number) or np.isinf(array.values).any():
            raise ValueError(f'{name} must contain numeric values or NaN tracking gaps.')

    valid_confidence = confidence.values[np.isfinite(confidence.values)]
    if np.any(valid_confidence < 0) or np.any(valid_confidence > 1):
        raise ValueError('confidence must lie between zero and one, or be NaN.')
    if fps is not None and (not np.isscalar(fps) or isinstance(fps, (str, bool)) or not np.isfinite(fps) or fps <= 0):
        raise ValueError('fps must be a positive finite number or None.')

    # === 3| Rename the Existing Dimensions and Add One Individual ================

    rename = {time_coord: 'time', 'keypoints': 'keypoint'}
    position = position.rename(rename).expand_dims({'individual': [individual]})
    confidence = confidence.rename(rename).expand_dims({'individual': [individual]})

    position = position.transpose('time', 'space', 'keypoint', 'individual')
    confidence = confidence.transpose('time', 'keypoint', 'individual')

    # === 4| Preserve Acquired Seconds and Return Independent Raw Pose ============

    dataset = xr.Dataset({'position': position, 'confidence': confidence})
    dataset.attrs['source_software'] = source_software
    dataset.attrs['time_unit'] = 'seconds'
    dataset.attrs['spatial_unit'] = 'pixels'
    dataset.time.attrs['units'] = 'seconds'
    dataset.position.attrs['units'] = 'pixels'
    if fps is not None:
        dataset.attrs['fps'] = float(fps)                     # FPS is descriptive metadata; the acquired time coordinate is unchanged.

    return dataset.copy(deep=True)                           # Processing this dataset cannot overwrite the original loaded arrays.

#===============================================================================



################################################################################





# '''
# DeepLabCut pose loading (frame-indexed) and alignment to camera frame times.
# ----------------------------------------------------------------------------



# Description:
#     DeepLabCut (DLC) writes one row of pose estimates per VIDEO FRAME, indexed
#     by frame number (0, 1, 2, ...). It carries no wall-clock / experiment time
#     of its own. The experiment clock for those frames lives elsewhere: the
#     Bonsai ``VideoData`` CSV logs one row per frame too, and its ``Seconds``
#     column is the HARP/Bonsai timestamp of each frame (the SAME clock that
#     ExperimentEvents, Nosepoke, SoundCard, etc. are on).

#     This module keeps those two jobs separate, by design:

#       * READING (a datasource).  ``DLCPose`` / ``read_dlc_pose`` read a session's
#         DLC output(s) into FRAME-INDEXED arrays. They know nothing about video or
#         time. A session split across several DLC files is concatenated in FILENAME
#         order and given one continuous ``0..N-1`` frame index (so the assembled
#         pose is monotonic, not a per-file reset). This is an ordinary source
#         object exposing ``.data_arrays``, just like the HARP presets.

#       * ALIGNING (a datastructure step).  ``align_pose_to_video`` takes a loaded
#         ``VideoData`` (or its flattened table) and a loaded ``DLCPose`` (or its
#         arrays), checks the two carry the SAME number of frames, and stamps the
#         video's per-frame ``Seconds`` onto the pose by position, returning
#         TIME-INDEXED arrays on the shared session clock. Because DLC is run per
#         video file, frame i of the concatenated pose is frame i of the
#         concatenated video, so the equal-length check is the alignment guarantee:
#         if the counts match, the positional join is sound.

#     Splitting the two responsibilities means there is ONE read of the VideoData
#     per session (the alignment reuses the already-loaded video rather than
#     re-reading the CSVs), the segment ordering is decided ONCE (filename order,
#     on both sides), and the frame -> time join lives at the datastructure level
#     where both streams are visible, instead of being buried inside the pose
#     reader.

#     Output shape (movement-friendly):
#       * ``position``   : xr.DataArray (frame|Time x keypoints x space[x, y])
#       * ``confidence`` : xr.DataArray (frame|Time x keypoints)
#     ``read_dlc_pose`` returns these on a ``frame`` axis; ``align_pose_to_video``
#     returns them on a ``Time`` axis. These two arrays are the natural split used
#     by the neuroinformatics ``movement`` package (position vs. likelihood);
#     ``pose_to_movement`` converts the pair into a ``movement``-schema
#     ``xr.Dataset`` when needed.

# Contents:
# --------------------------------
# - read_dlc_pose:       Load one session's DLC pose as frame-indexed arrays.
# - DLCPose:             Source-style wrapper exposing ``.data_arrays`` for the catalog.
# - align_pose_to_video: Stamp VideoData frame times onto frame-indexed pose (with checks).
# - pose_to_movement:    Convert (position, confidence) into a movement-schema Dataset.
# '''





# ################################################################################
# # Imports
# ################################################################################

# import warnings
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import xarray as xr

# from data_conduit.core.utils.utils_core import _concat_split_dataframes

# ################################################################################



# # Constant: the three DLC per-keypoint sub-columns. ``x``/``y`` make up the
# # spatial position; ``likelihood`` is the detection confidence kept separately.
# _SPACE_COORDS = ('x', 'y')
# _LIKELIHOOD_COORD = 'likelihood'




# ################################################################################
# # Private Helpers
# ################################################################################



# #===============================================================================
# # 1| List the DLC Output Files in a Session
# #===============================================================================
# def _dlc_files(
#         dlc_dir: Path,
#         file_format: str,
# ) -> list[Path]:
#     '''
#     Return the DLC output files in ``dlc_dir`` for the requested format.

#     Prefers ``.csv`` and falls back to ``.h5`` when no CSV files are present,
#     unless a format is pinned explicitly. DLC also
#     writes ``*_meta.pickle`` / ``*_full.pickle`` sidecars; those are ignored.

#     ----------
#     Parameters:
#         dlc_dir (Path):
#             The session's ``DLC`` subdirectory.
#         file_format (str):
#             ``'csv'``, ``'h5'``, or ``'auto'`` (csv if any, else h5).
#     Returns:
#         list[Path]:
#             Matching DLC files, sorted by name (so a multi-file session is read
#             in filename, hence chronological, order).
#     '''

#     # Resolve 'auto' to whichever format actually has files present, preferring
#     # CSV because that is the lab's canonical DLC export for this workflow.
#     if file_format == 'auto':
#         file_format = 'csv' if list(dlc_dir.glob('*.csv')) else 'h5'

#     if file_format not in ('h5', 'csv'):
#         raise ValueError(f"file_format must be 'h5', 'csv', or 'auto' (got {file_format!r}).")

#     return sorted(dlc_dir.glob(f'*.{file_format}'))

# #===============================================================================



# #===============================================================================
# # 2| Read One DLC Table (h5 or csv) Into a 2-Level-Column DataFrame
# #===============================================================================
# def _read_dlc_table(
#         path: Path,
# ) -> pd.DataFrame:
#     '''
#     Read one DLC file into a DataFrame with (bodyparts, coords) columns.

#     DLC files carry a 3-level column header (scorer, bodyparts, coords) and a
#     frame-number index. The single-valued ``scorer`` level is dropped so the
#     columns are just ``(bodypart, {x, y, likelihood})``.

#     ----------
#     Parameters:
#         path (Path):
#             A DLC ``.h5`` or ``.csv`` output file.
#     Returns:
#         pd.DataFrame:
#             One row per frame; columns are a 2-level MultiIndex
#             ``(bodyparts, coords)``.
#     '''

#     # h5 stores the frame index + 3-level columns natively; the csv carries 3
#     # header rows (scorer / bodyparts / coords) with the frame index in column 0.
#     df = pd.read_hdf(path) if path.suffix == '.h5' else pd.read_csv(path, header=[0, 1, 2], index_col=0)

#     # Drop the scorer level if present (single model -> single scorer).
#     if df.columns.nlevels == 3:
#         df.columns = df.columns.droplevel(0)
#     df.columns = df.columns.set_names(['bodyparts', 'coords'])
#     return df

# #===============================================================================



# #===============================================================================
# # 3| Turn One DLC Table Into Frame-Indexed position / confidence Arrays
# #===============================================================================
# def _table_to_arrays(
#         table: pd.DataFrame,
#         frame_dim: str,
# ) -> tuple[xr.DataArray, xr.DataArray]:
#     '''
#     Turn one DLC table into frame-indexed ``position`` + ``confidence`` arrays.

#     No times are attached here. The arrays carry a bare ``frame_dim`` axis with
#     NO coordinate; a continuous frame index is assigned once in
#     ``read_dlc_pose`` after all of a session's files are concatenated.

#     ----------
#     Parameters:
#         table (pd.DataFrame):
#             One DLC file's table with ``(bodyparts, coords)`` columns.
#         frame_dim (str):
#             Name to give the per-frame dimension (e.g. ``'frame'``).
#     Returns:
#         tuple[xr.DataArray, xr.DataArray]:
#             ``(position, confidence)`` for this file, where ``position`` is
#             ``(frame x keypoints x space)`` and ``confidence`` is
#             ``(frame x keypoints)``.
#     '''

#     # Split out x / y / likelihood; each is (frames x bodyparts). Reindex y and
#     # likelihood to x's bodypart order so the stack lines up regardless of the
#     # column order DLC happened to write.
#     xs = table.xs('x', level='coords', axis=1)
#     keypoints = list(xs.columns)
#     ys = table.xs('y', level='coords', axis=1)[keypoints]
#     likelihood = table.xs(_LIKELIHOOD_COORD, level='coords', axis=1)[keypoints]

#     # position: stack x and y along a new 'space' axis -> (frames, keypoints, 2).
#     # We deliberately leave frame_dim coordinate-free; it is filled after concat.
#     position_values = np.stack([xs.to_numpy(dtype=float), ys.to_numpy(dtype=float)], axis=-1)
#     position = xr.DataArray(
#         position_values,
#         dims=(frame_dim, 'keypoints', 'space'),
#         coords={'keypoints': keypoints, 'space': list(_SPACE_COORDS)},
#         name='position',
#     )

#     confidence = xr.DataArray(
#         likelihood.to_numpy(dtype=float),
#         dims=(frame_dim, 'keypoints'),
#         coords={'keypoints': keypoints},
#         name='confidence',
#     )
#     return position, confidence

# #===============================================================================



# #===============================================================================
# # 4| Extract Per-Frame Video Times From a VideoData Object / Table
# #===============================================================================
# def _video_times(
#         video,
# ) -> np.ndarray:
#     '''
#     Pull the per-frame camera times out of a VideoData object (or table).

#     Accepts whatever the caller has to hand: a ``VideoData`` source object (its
#     ``.df`` is used), a pre-flattened DataFrame (its ``Time`` index is the
#     per-frame clock, because VideoData is read with the ``Seconds`` column as
#     ``index_col=0`` and renamed to ``Time``), a Series, or a bare array of
#     times. The result is the one-value-per-frame timestamp vector the pose is
#     aligned to.

#     Only a raw ``VideoData`` object is flattened here (its ``.df`` may be a
#     multi-file dict), using the filename-first policy. Anything already flattened
#     (e.g. a loaded ``video`` bundle member) is trusted in the order given and is
#     NOT re-sorted: re-sorting could mask a camera-clock reset that the flattening
#     deliberately left visible as a non-monotonic index.

#     ----------
#     Parameters:
#         video (object | pandas.DataFrame | pandas.Series | array-like):
#             A ``VideoData`` source object, its flattened table, or a times array.
#     Returns:
#         numpy.ndarray:
#             1D float array of per-frame times, one per camera frame.
#     '''

#     # A VideoData source object carries the table on ``.df`` and may be split
#     # across files (a dict), so flatten it with the shared filename-first policy.
#     # on_rollback='ignore' because align_pose_to_video does its own (configurable)
#     # rollback check on the final Time axis, so we do not want a second warning here.
#     # Anything else is assumed to be ALREADY flattened, so we trust its order.
#     if hasattr(video, 'df'):
#         table = _concat_split_dataframes(video.df, on_rollback='ignore')
#     else:
#         table = video

#     # The per-frame clock is the Time index for a DataFrame, the values for a
#     # Series, or the array itself otherwise.
#     if isinstance(table, pd.DataFrame):
#         return table.index.to_numpy(dtype=float)
#     if isinstance(table, pd.Series):
#         return table.to_numpy(dtype=float)
#     return np.asarray(table, dtype=float)

# #===============================================================================



# ################################################################################




# ################################################################################
# # Public API
# ################################################################################



# #===============================================================================
# # 1| read_dlc_pose (Load One Session's Pose as Frame-Indexed Arrays)
# #===============================================================================
# def read_dlc_pose(
#         experiment_directory_path: str | Path,
#         *,
#         dlc_subdir: str = 'DLC',
#         file_format: str = 'csv',
#         frame_dim: str = 'frame',
# ) -> dict[str, xr.DataArray]:
#     '''
#     Load one session's DLC pose into frame-indexed arrays (no times attached).

#     Reads every DLC output in ``<session>/<dlc_subdir>`` in filename order,
#     concatenates them, and gives the whole session one continuous ``0..N-1``
#     frame index. This is a pure reader: it does not look at VideoData and does
#     not put pose on the session clock. Use ``align_pose_to_video`` for that.

#     ----------
#     Parameters:
#         experiment_directory_path (str | Path):
#             One session directory (the folder that contains the ``DLC``
#             subfolder).
#         dlc_subdir (str):
#             Name of the DLC output subfolder. Default ``'DLC'``.
#         file_format (str):
#             ``'csv'`` (default), ``'h5'``, or ``'auto'`` (csv if present, else h5).
#         frame_dim (str):
#             Name to give the per-frame dimension. Default ``'frame'``.
#     Returns:
#         dict[str, xr.DataArray]:
#             ``{'position': (frame x keypoints x space), 'confidence':
#             (frame x keypoints)}``, with ``frame_dim`` a continuous ``0..N-1``
#             index spanning all of the session's DLC files.
#     '''

#     session = Path(experiment_directory_path)
#     dlc_dir = session / dlc_subdir

#     if not dlc_dir.is_dir():
#         raise FileNotFoundError(f'no {dlc_subdir!r} folder in session {session}.')

#     dlc_paths = _dlc_files(dlc_dir, file_format)
#     if not dlc_paths:
#         raise FileNotFoundError(f'no DLC {file_format!r} files in {dlc_dir}.')

#     # 1| Read each DLC file (filename order) into frame-indexed arrays.
#     positions, confidences = [], []
#     for path in dlc_paths:
#         table = _read_dlc_table(path)
#         position, confidence = _table_to_arrays(table, frame_dim)
#         positions.append(position)
#         confidences.append(confidence)

#     # 2| Concatenate the files along the frame axis (a no-op for a single file).
#     position = positions[0] if len(positions) == 1 else xr.concat(positions, dim=frame_dim)
#     confidence = confidences[0] if len(confidences) == 1 else xr.concat(confidences, dim=frame_dim)

#     # 3| Give the assembled pose one continuous 0..N-1 frame index, so it is
#     #    monotonic rather than restarting at 0 per file. This is the index the
#     #    aligner replaces with the matching video Times.
#     frame_index = np.arange(position.sizes[frame_dim])
#     position = position.assign_coords({frame_dim: frame_index})
#     confidence = confidence.assign_coords({frame_dim: frame_index})

#     return {'position': position, 'confidence': confidence}

# #===============================================================================



# #===============================================================================
# # 2| DLCPose (Source-Style Wrapper for the DataStructureCatalog)
# #===============================================================================
# class DLCPose:
#     '''
#     Thin source-style wrapper around ``read_dlc_pose``.

#     Exposes the loaded pose as a ``.data_arrays`` mapping, the same interface
#     the HARP presets (Nosepoke, SoundCard, ...) expose, so it drops straight
#     into a ``DataStructureCatalog`` spec::

#         catalog.add(DataStructureSpec(
#             name='dlc',
#             reader=lambda p: DLCPose(experiment_directory_path=p),
#         ))

#     and the loader turns its two arrays into the bundle members
#     ``'dlc:position'`` and ``'dlc:confidence'``. NOTE that those members are
#     FRAME-INDEXED: this wrapper does not attach camera times. To put pose on the
#     session clock, pass this object (or its arrays) together with the session's
#     ``VideoData`` to ``align_pose_to_video``.

#     ----------
#     Parameters:
#         experiment_directory_path (str | Path):
#             One session directory. Required.
#         **kwargs:
#             Forwarded to ``read_dlc_pose`` (``dlc_subdir``, ``file_format``,
#             ``frame_dim``).

#     Attributes:
#         experiment_directory_path (Path):
#             The session directory this pose was read from.
#         data_arrays (dict[str, xr.DataArray]):
#             ``{'position': ..., 'confidence': ...}``, frame-indexed.
#     '''

#     def __init__(
#             self,
#             experiment_directory_path: str | Path | None = None,
#             **kwargs,
#     ) -> None:
#         '''
#         Load the session's frame-indexed pose into ``self.data_arrays``.

#         ----------
#         Parameters:
#             experiment_directory_path (str | Path | None):
#                 One session directory. Required (None raises).
#             **kwargs:
#                 Forwarded to ``read_dlc_pose``.
#         Returns:
#             None.
#         '''
#         if experiment_directory_path is None:
#             raise ValueError('experiment_directory_path is required.')
#         self.experiment_directory_path = Path(experiment_directory_path)
#         self.data_arrays = read_dlc_pose(self.experiment_directory_path, **kwargs)

# #===============================================================================



# #===============================================================================
# # 3| align_pose_to_video (Stamp VideoData Times Onto Frame-Indexed Pose)
# #===============================================================================
# def align_pose_to_video(
#         video,
#         pose,
#         *,
#         frame_dim: str = 'frame',
#         time_coord: str = 'Time',
#         on_rollback: str = 'warn',
# ) -> dict[str, xr.DataArray]:
#     '''
#     Put frame-indexed pose on the session clock using the VideoData frame times.

#     This is the datastructure-level join that the datasource deliberately does
#     NOT do. It takes a session's loaded ``VideoData`` and ``DLCPose`` (the two
#     streams of camera frames), checks they have the SAME number of frames, and
#     stamps the video's per-frame time onto the pose by position. Because DLC is
#     run per video file, frame i of the (filename-ordered) pose is frame i of the
#     (filename-ordered) video, so equal total length is the alignment guarantee.

#     ----------
#     Parameters:
#         video (object | pandas.DataFrame | pandas.Series | array-like):
#             The session's camera frame times. Either a ``VideoData`` source
#             object (its ``.df`` is used), its flattened ``Time``-indexed table,
#             or a bare array of per-frame times. Whatever is given is flattened
#             in filename order and read as one time-per-frame vector.
#         pose (object | dict[str, xr.DataArray]):
#             The session's frame-indexed pose. Either a ``DLCPose`` object (its
#             ``.data_arrays`` is used) or a ``{'position': ..., 'confidence':
#             ...}`` mapping, both on a ``frame_dim`` axis.
#         frame_dim (str):
#             Name of the pose's per-frame dimension to convert. Default
#             ``'frame'`` (matches ``read_dlc_pose``).
#         time_coord (str):
#             Name to give the resulting time dimension. Default ``'Time'``
#             (matches the sessiongroups alignment pipeline).
#         on_rollback (str):
#             What to do when the resulting time axis is not increasing (the
#             camera clock reset between video files): ``'warn'`` (default) to
#             warn and keep the data in filename order, ``'error'`` to raise, or
#             ``'ignore'`` to do neither.
#     Returns:
#         dict[str, xr.DataArray]:
#             ``{'position': (Time x keypoints x space), 'confidence':
#             (Time x keypoints)}`` on the shared session clock.
#     '''

#     # 1| Per-frame video times (flattened, filename order, one value per frame).
#     times = _video_times(video)

#     # 2| Frame-indexed pose arrays (accept a DLCPose object or a bare dict).
#     arrays = pose.data_arrays if hasattr(pose, 'data_arrays') else pose
#     position, confidence = arrays['position'], arrays['confidence']

#     # 3| The equal-entries check: this IS the alignment guarantee. If the pose
#     #    and the video carry the same number of frames, the positional join is
#     #    sound; if not, we cannot align them, so fail loudly with the counts.
#     n_pose = position.sizes[frame_dim]
#     if n_pose != len(times):
#         raise ValueError(
#             f'cannot align pose to video: pose has {n_pose} frames but VideoData has '
#             f'{len(times)} frame times. DLC is run per video file, so the two must '
#             f'correspond one-to-one; a mismatch points to a dropped frame, a '
#             f'wrong/stale DLC file, or a missing VideoData segment.'
#         )

#     # 4| Stamp the times onto the pose by POSITION: rename the frame axis to the
#     #    time axis and attach the per-frame times as its coordinate. Both arrays
#     #    share the frame axis, so the same times land on both.
#     position = position.rename({frame_dim: time_coord}).assign_coords({time_coord: times})
#     confidence = confidence.rename({frame_dim: time_coord}).assign_coords({time_coord: times})

#     # 5| Surface a camera-clock reset: ordered by filename, the only way the time
#     #    axis can step backwards is an actual rollback between video files. We
#     #    leave the data in filename order (not time-sorted) so the reset stays
#     #    visible, and announce it per the caller's policy.
#     if on_rollback != 'ignore' and np.any(np.diff(times) < 0):
#         message = (
#             'aligned pose Time axis steps backwards; the camera clock appears to '
#             'have reset between VideoData files. Pose is left in filename order '
#             '(not time-sorted) so the reset stays visible; downstream steps '
#             'assuming monotonic time may need attention.'
#         )
#         if on_rollback == 'error':
#             raise ValueError(message)
#         warnings.warn(message, stacklevel=2)

#     return {'position': position, 'confidence': confidence}

# #===============================================================================



# #===============================================================================
# # 4| pose_to_movement (Convert to a movement-Schema Dataset)
# #===============================================================================
# def pose_to_movement(
#         position: xr.DataArray,
#         confidence: xr.DataArray,
#         *,
#         individual: str = 'individual_0',
#         time_coord: str = 'Time',
#         source_software: str = 'DeepLabCut',
#         fps: float | None = None,
# ):
#     '''
#     Pack ``position`` + ``confidence`` into a ``movement``-schema ``xr.Dataset``.

#     The neuroinformatics ``movement`` package expects a poses dataset with
#     dims ``(time, individuals, keypoints, space)`` and data variables
#     ``position`` and ``confidence``. data-conduit keeps the arrays on a
#     ``Time`` axis (and possibly with extra coords such as the session
#     ``label`` after a combine); this renames the time dim, adds the singleton
#     ``individuals`` dimension, orders the dims, and sets the attributes
#     ``movement`` looks for. Any extra coordinates riding the time axis (e.g.
#     ``label``) are preserved.

#     ----------
#     Parameters:
#         position (xr.DataArray):
#             ``(Time x keypoints x space)`` positions.
#         confidence (xr.DataArray):
#             ``(Time x keypoints)`` likelihoods.
#         individual (str):
#             Name for the single tracked individual. Default ``'individual_0'``.
#         time_coord (str):
#             Current name of the time dimension to rename to ``'time'``.
#             Default ``'Time'``.
#         source_software (str):
#             Value for the dataset's ``source_software`` attribute.
#         fps (float | None):
#             Optional frames-per-second to record in ``attrs``.
#     Returns:
#         xr.Dataset:
#             A ``movement``-compatible poses dataset.
#     '''

#     rename = {time_coord: 'time'} if time_coord in position.dims else {}
#     position = position.rename(rename).expand_dims({'individuals': [individual]})
#     confidence = confidence.rename(rename).expand_dims({'individuals': [individual]})

#     position = position.transpose('time', 'individuals', 'keypoints', 'space', ...)
#     confidence = confidence.transpose('time', 'individuals', 'keypoints', ...)

#     dataset = xr.Dataset({'position': position, 'confidence': confidence})
#     dataset.attrs['source_software'] = source_software
#     if fps is not None:
#         dataset.attrs['fps'] = fps
#     return dataset

# #===============================================================================



# ################################################################################

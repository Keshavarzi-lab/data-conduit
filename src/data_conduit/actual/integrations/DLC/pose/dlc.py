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
        TIME-INDEXED arrays on the shared session clock. Because DLC is run per
        video file, frame i of the concatenated pose is frame i of the
        concatenated video, so the equal-length check is the alignment guarantee:
        if the counts match, the positional join is sound.

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

from data_conduit.actual.core.utils.utils_core import _concat_split_dataframes

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
    multi-file dict), using the filename-first policy. Anything already flattened
    (e.g. a loaded ``video`` bundle member) is trusted in the order given and is
    NOT re-sorted: re-sorting could mask a camera-clock reset that the flattening
    deliberately left visible as a non-monotonic index.

    ----------
    Parameters:
        video (object | pandas.DataFrame | pandas.Series | array-like):
            A ``VideoData`` source object, its flattened table, or a times array.
    Returns:
        numpy.ndarray:
            1D float array of per-frame times, one per camera frame.
    '''

    # A VideoData source object carries the table on ``.df`` and may be split
    # across files (a dict), so flatten it with the shared filename-first policy.
    # on_rollback='ignore' because align_pose_to_video does its own (configurable)
    # rollback check on the final Time axis, so we do not want a second warning here.
    # Anything else is assumed to be ALREADY flattened, so we trust its order.
    if hasattr(video, 'df'):
        table = _concat_split_dataframes(video.df, on_rollback='ignore')
    else:
        table = video

    # The per-frame clock is the Time index for a DataFrame, the values for a
    # Series, or the array itself otherwise.
    if isinstance(table, pd.DataFrame):
        return table.index.to_numpy(dtype=float)
    if isinstance(table, pd.Series):
        return table.to_numpy(dtype=float)
    return np.asarray(table, dtype=float)

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
    stamps the video's per-frame time onto the pose by position. Because DLC is
    run per video file, frame i of the (filename-ordered) pose is frame i of the
    (filename-ordered) video, so equal total length is the alignment guarantee.

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
            camera clock reset between video files): ``'warn'`` (default) to
            warn and keep the data in filename order, ``'error'`` to raise, or
            ``'ignore'`` to do neither.
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

    # 3| The equal-entries check: this IS the alignment guarantee. If the pose
    #    and the video carry the same number of frames, the positional join is
    #    sound; if not, we cannot align them, so fail loudly with the counts.
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

    # 5| Surface a camera-clock reset: ordered by filename, the only way the time
    #    axis can step backwards is an actual rollback between video files. We
    #    leave the data in filename order (not time-sorted) so the reset stays
    #    visible, and announce it per the caller's policy.
    if on_rollback != 'ignore' and np.any(np.diff(times) < 0):
        message = (
            'aligned pose Time axis steps backwards; the camera clock appears to '
            'have reset between VideoData files. Pose is left in filename order '
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

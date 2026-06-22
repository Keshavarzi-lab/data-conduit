'''
Pose-estimation sources (DeepLabCut), aligned to camera frame times.
--------------------------------------------------------------------

Description:
    Pose data (DeepLabCut) is logged per video frame with no clock of its own.
    These readers attach the camera's per-frame HARP/Bonsai timestamps (from
    the Bonsai ``VideoData`` CSV) to the pose rows, putting pose on the shared
    session clock so it combines with the rest of a session via the
    sessiongroups pipeline. See ``dlc.py`` for the full rationale.

Contents:
--------------------------------
- DLCPose:          Source-style wrapper exposing ``.data_arrays`` for the catalog.
- read_dlc_pose:    Load + frame-time-align one session's DLC pose.
- pose_to_movement: Convert (position, confidence) into a movement-schema Dataset.
'''

from data_conduit.datasources.pose.dlc import DLCPose, pose_to_movement, read_dlc_pose

__all__ = [
    'DLCPose',
    'read_dlc_pose',
    'pose_to_movement',
]

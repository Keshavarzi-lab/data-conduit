'''
Pose-estimation sources (DeepLabCut) and alignment to camera frame times.
-------------------------------------------------------------------------

Description:
    Pose data (DeepLabCut) is logged per video frame with no clock of its own.
    Reading and time-alignment are kept separate (see ``dlc.py`` for the full
    rationale):

      * ``DLCPose`` / ``read_dlc_pose`` read a session's DLC output(s) into
        FRAME-INDEXED arrays (an ordinary datasource, no video involved).
      * ``align_pose_to_video`` stamps the camera's per-frame HARP/Bonsai
        timestamps (from the Bonsai ``VideoData`` CSV) onto the pose, putting it
        on the shared session clock so it combines with the rest of a session.

Contents:
--------------------------------
- DLCPose:             Source-style wrapper exposing ``.data_arrays`` for the catalog.
- read_dlc_pose:       Load one session's DLC pose as frame-indexed arrays.
- align_pose_to_video: Stamp VideoData frame times onto frame-indexed pose (with checks).
- pose_to_movement:    Convert (position, confidence) into a movement-schema Dataset.
'''

from data_conduit.datasources.pose.dlc import (
    DLCPose,
    align_pose_to_video,
    pose_to_movement,
    read_dlc_pose,
)

__all__ = [
    'DLCPose',
    'read_dlc_pose',
    'align_pose_to_video',
    'pose_to_movement',
]

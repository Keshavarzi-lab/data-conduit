'''
Workspace paths and a figure saver for the pose-trajectory notebooks.
=====================================================================

Description:
    Small, boring infrastructure shared by every notebook: where this workspace
    lives on disk, where to drop saved tables and figures, a consistent figure
    saver, and a helper to sort "Day 1", "Day 10", "Day 2" into numeric order.

    Nothing here touches experiment data or ``movement``; it only knows about
    folders and files. Analysis knobs (which keypoints, confidence thresholds,
    and so on) deliberately do NOT live here: those are passed at the call site in
    the notebooks so each analysis states its own choices explicitly.

Contents:
--------------------------------
- REPO_ROOT:        Absolute path to the data-conduit repository root.
- WORKSPACE_ROOT:   Absolute path to the Pose_Trajectory_Analysis folder.
- OUTPUT_DIR:       Where saved tables (summary table, poke positions) are written.
- FIGURE_DIR:       Where saved figures are written.
- day_order:        Sort key turning a day-folder name into its integer.
- dated_name:       Append today's date to a file stem.
- save_figure:      Save a matplotlib figure under FIGURE_DIR and return its path.
'''

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

# REPO_ROOT is resolved once in the package __init__; reuse it (and re-export it via __all__
# below) so there is one definition of "where is the repo" for the package and the notebooks.
from pose_trajectory import REPO_ROOT

# === Paths ===================================================================
# The workspace folder (this package's parent's parent: pose_trajectory/ -> the
# Pose_Trajectory_Analysis/ folder). Outputs and figures live beside the notebooks
# so the whole analysis is self-contained in one directory.
WORKSPACE_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Path = WORKSPACE_ROOT / 'outputs'
FIGURE_DIR: Path = WORKSPACE_ROOT / 'figures'

# Re-exported so notebooks reach these as ``workspace.<name>``. Listing REPO_ROOT here also
# tells the linter the import above is an intentional re-export, not dead code.
__all__ = ['REPO_ROOT', 'WORKSPACE_ROOT', 'OUTPUT_DIR', 'FIGURE_DIR', 'day_order', 'dated_name', 'save_figure']


def day_order(day: str) -> int:
    '''
    Sort key for day-folder names so they order numerically, not lexically.

    Folder names like ``"Day 1"``, ``"Day 10"``, ``"Day 2"`` sort wrongly as plain
    strings ("Day 10" before "Day 2"). This returns the first run of digits in the
    name as an integer, so they order 1, 2, 10. Names with no digit sort to the end.

    ----------
    Parameters:
        day (str):
            A day-folder name, e.g. ``"Day 1"`` or ``"Day_11"``.
    Returns:
        int:
            The first integer found in the name, or a large sentinel (10000) when
            the name contains no digits so such names sort last.
    '''
    # Grab the first group of consecutive digits anywhere in the string.
    match = re.search(r'(\d+)', str(day))
    # Fall back to a big number so unparseable names land at the end of a sort.
    return int(match.group(1)) if match else 10_000


def dated_name(stem: str, suffix: str) -> str:
    '''
    Build a filename of the form ``"{stem}_{YYYY-MM-DD}{suffix}"``.

    Used so repeated runs on different days do not silently overwrite each other and
    so a saved file carries the date it was produced.

    ----------
    Parameters:
        stem (str):
            The base name, e.g. ``"summary_table"``.
        suffix (str):
            The file extension including its dot, e.g. ``".csv"`` or ``".png"``.
    Returns:
        str:
            The dated filename, e.g. ``"summary_table_2026-06-26.csv"``.
    '''
    # ISO date keeps files sorting chronologically by name.
    today = _dt.date.today().isoformat()
    return f'{stem}_{today}{suffix}'


def save_figure(fig, name: str, *, dated: bool = True, dpi: int = 150) -> Path:
    '''
    Save a matplotlib figure under FIGURE_DIR and return the written path.

    Creates FIGURE_DIR if needed. By default the date is appended to the name (see
    ``dated_name``); pass ``dated=False`` for a stable, overwrite-in-place name.

    ----------
    Parameters:
        fig (matplotlib.figure.Figure):
            The figure to save.
        name (str):
            File stem, with or without an extension. If no extension is given,
            ``.png`` is used.
        dated (bool):
            If True (default), append today's date to the stem via ``dated_name``.
        dpi (int):
            Resolution in dots per inch for raster formats. Default 150.
    Returns:
        pathlib.Path:
            The absolute path the figure was written to.
    '''
    # Make sure the output folder exists before writing into it.
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Split the caller's name into a stem and an extension, defaulting to .png so a
    # bare "speed_over_time" becomes "speed_over_time.png".
    stem, dot, ext = name.rpartition('.')
    if dot:
        suffix = f'.{ext}'
    else:
        stem, suffix = name, '.png'

    # Optionally stamp the date into the filename.
    filename = dated_name(stem, suffix) if dated else f'{stem}{suffix}'
    path = FIGURE_DIR / filename

    # bbox_inches='tight' trims surrounding whitespace so saved panels crop cleanly.
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    return path

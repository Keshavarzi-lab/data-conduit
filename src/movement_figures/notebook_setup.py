"""Refresh local project imports when a notebook is rerun after source changes.

Notebook import cells load this file by path with ``runpy.run_path``. That makes
the refresh available even if an older movement_figures package is cached.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


# ============================================================================
# 1 | Select the Checkout and Refresh Its Project Modules
# ============================================================================

def refresh_project_imports(source_root: str | Path) -> Path:
    """Make subsequent project imports read the selected checkout again.

    Parameters
    ----------
    source_root : str or pathlib.Path
        Directory containing both ``data_conduit/`` and ``movement_figures/``.
        Notebook import cells find this directory from their working directory.

    Returns
    -------
    pathlib.Path
        Resolved source directory placed first on Python's import search path.
        Later imports load both project packages from their current source.

    Notes
    -----
    Cached modules in these two project packages are removed together so the
    loader, catalog and data structures cannot remain on different revisions.
    Third-party modules are retained. Existing notebook variables are not
    rewritten: rerun the following cells to rebuild data and figures after
    running this import cell, as with a normal top-to-bottom notebook execution.
    This function does not edit source files, install packages or change data.

    Raises
    ------
    FileNotFoundError
        The directory does not contain both required project packages.
    """
    # --- 1.1 | Validate the checkout before touching the import cache -----------
    source_root = Path(source_root).resolve()
    package_names = ("data_conduit", "movement_figures")
    if not all((source_root / name).is_dir() for name in package_names):
        raise FileNotFoundError(
            f"Expected data_conduit and movement_figures directories in {source_root}."
        )

    # --- 1.2 | Give this checkout precedence over editable or older installs ----
    source_path = str(source_root)
    sys.path[:] = [entry for entry in sys.path if entry != source_path]
    sys.path.insert(0, source_path)

    # --- 1.3 | Refresh the complete project graph, including cached parents -----
    # Reloading only loading.py could leave its imported catalog/classes stale.
    cached_names = [
        name for name in sys.modules
        if any(name == package or name.startswith(package + ".") for package in package_names)
    ]
    for name in sorted(cached_names, key=lambda value: value.count("."), reverse=True):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return source_root

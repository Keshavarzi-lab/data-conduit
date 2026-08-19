"""Tools for constructing reusable, multi-stream experimental data pipelines."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    __version__ = _distribution_version("data-conduit")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]

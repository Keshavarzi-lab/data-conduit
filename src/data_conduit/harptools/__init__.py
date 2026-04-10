'''Harptools module for Data Conduit.'''

try:
    import harp  # noqa - check harp-python is available
except ImportError:
    raise ImportError(
        "harp-python is required to use harptools. "
        "Install it with: pip install harp-python"
    )

from data_conduit.harptools.harptools_core import (
    construct_device_reader,
    collect_registers,
    read_harp_bin,
    collect_harp_dfs,
)

from data_conduit.io import add_reader
add_reader("harp_bin", read_harp_bin)

__all__ = [
    "construct_device_reader",
    "collect_registers",
    "read_harp_bin",
    "collect_harp_dfs",
]




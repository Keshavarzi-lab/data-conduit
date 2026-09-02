"""Side-effect-free helpers for reading HARP binary data."""

from ....core.io import add_reader, get_reader
from .harptools_core import collect_harp_dfs, collect_registers, construct_device_reader, read_harp_bin


def register_harp_reader() -> None:
    """Register the ``harp_bin`` reader with the shared IO registry."""
    try:
        registered_reader = get_reader("harp_bin")
    except KeyError:
        add_reader("harp_bin", read_harp_bin)
        return

    if registered_reader is read_harp_bin:
        return

    raise ValueError("A different reader is already registered as 'harp_bin'.")


__all__ = [
    "collect_harp_dfs",
    "collect_registers",
    "construct_device_reader",
    "read_harp_bin",
    "register_harp_reader",
]

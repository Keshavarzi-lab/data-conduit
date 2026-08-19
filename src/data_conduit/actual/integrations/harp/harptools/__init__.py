'''Harptools module for Data Conduit.

Importing this package is side-effect free and does NOT require harp-python:
the ``harp`` dependency is pulled in lazily by ``construct_device_reader`` only
when a real reader is built, and the ``harp_bin`` reader is registered on the IO
registry explicitly via ``register_harp_reader()`` rather than at import time.
'''

from data_conduit.actual.core.io import add_reader
from data_conduit.actual.integrations.harp.harptools.harptools_core import (
    collect_harp_dfs,
    collect_registers,
    construct_device_reader,
    read_harp_bin,
)


def register_harp_reader() -> None:
    '''Register the ``harp_bin`` reader on the IO registry.

    Kept out of import scope so importing this package mutates no global state.
    Callers (or the HARP datasource presets) invoke this explicitly when they
    want ``collect_dfs`` to recognise ``.bin`` HARP files.
    '''
    add_reader("harp_bin", read_harp_bin)


__all__ = [
    "collect_harp_dfs",
    "collect_registers",
    "construct_device_reader",
    "read_harp_bin",
    "register_harp_reader",
]

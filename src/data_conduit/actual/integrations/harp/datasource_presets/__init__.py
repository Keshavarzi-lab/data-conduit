"""
HARP preset subpackage for data-conduit.

Provides Device/MultiDevice wrappers and named presets for HARP-style binary
data. Requires ``harp-python`` to be installed.
"""

try:
    from data_conduit.actual.integrations.harp.datasource_presets.devices import (
        Device,
        SoundCard,
        CameraStart,
        Camera0Frames,
    )
    from data_conduit.actual.integrations.harp.datasource_presets.multidevice import (
        MultiDevice,
        Nosepoke,
    )
except ImportError as e:
    raise ImportError(
        "harp-python is required for HARP presets. "
        "Install it with: pip install harp-python"
    ) from e

__all__ = [
    "Device",
    "SoundCard",
    "CameraStart",
    "Camera0Frames",
    "MultiDevice",
    "Nosepoke",
]

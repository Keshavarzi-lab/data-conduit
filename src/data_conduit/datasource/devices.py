"""
Device wrappers built on top of DataSource.
------------------------------------------

Description:
    This module provides a minimal device layer for HARP-style data.
    It keeps DataSource generic and adds only the device-specific steps
    needed to load .bin files, reshape the resulting dfs_dict into a
    device/register layout, optionally validate expected content, and
    expose named DataArrays.

Contents:
    - Device
        Minimal HARP-style device wrapper around DataSource.
    - SoundCard
        Preset for SoundCard register access.
    - CameraStart
        Preset for camera start register access.
    - Camera0Frames
        Preset for camera frame register access, with optional matching-time
        filtering helper.
    - AnalogSync
        Preset for analog sync register access, with optional matching-time
        filtering helper.
"""

################################################################################
# Imports
################################################################################

from pathlib import Path

import xarray as xr

from data_conduit.datasource.datasource_core import DataSource, _build_data_arrays
from data_conduit.harptools import read_harp_bin
from data_conduit.utils import starts_with

################################################################################


def _collapse_device_folder_level(dfs_dict: dict) -> dict:
    """Collapse a duplicated device folder level in the loaded tree."""
    collapsed = {}

    for device_name, value in dfs_dict.items():
        if isinstance(value, dict) and list(value.keys()) == [device_name]:
            collapsed[device_name] = value[device_name]
        else:
            collapsed[device_name] = value

    return collapsed


def _rekey_harp_registers(dfs_dict: dict) -> dict:
    """
    Rekey HARP file stems to register keys.

    Converts keys such as ``Behavior0_8_timestamp`` into ``8`` so that the
    device tree becomes ``device -> register -> dataframe``.

    Raises
    ------
    ValueError
        If two source keys for the same device would collapse to the same
        register key and overwrite one another.
    """
    rekeyed_devices = {}

    for device_name, value in dfs_dict.items():
        if not isinstance(value, dict):
            rekeyed_devices[device_name] = value
            continue

        register_dict = {}
        for key, item in value.items():
            parts = key.split("_")
            register_key = parts[1] if len(parts) > 1 else key
            if register_key in register_dict:
                raise ValueError(
                    f"Rekeying would overwrite data for device '{device_name}' at register "
                    f"'{register_key}'. Conflicting source keys include '{key}'."
                )
            register_dict[register_key] = item
        rekeyed_devices[device_name] = register_dict

    return rekeyed_devices


def _filter_device_registers(dfs_dict: dict, device_registers: dict | None) -> dict:
    """Keep only requested registers for devices listed in device_registers."""
    if device_registers is None:
        return dfs_dict

    filtered = {}
    for device_name, registers in dfs_dict.items():
        if device_name not in device_registers:
            filtered[device_name] = registers
            continue

        allowed = {str(register) for register in device_registers[device_name]}
        if isinstance(registers, dict):
            filtered[device_name] = {
                register_name: item
                for register_name, item in registers.items()
                if register_name in allowed
            }
        else:
            filtered[device_name] = registers

    return filtered


def _warn_missing_expected_devices(dfs_dict: dict, device_list: list[str] | None, verbose: bool) -> None:
    """Print warnings for expected devices that were not loaded."""
    if not verbose or not device_list:
        return

    missing = [device_name for device_name in device_list if device_name not in dfs_dict]
    for device_name in missing:
        print(f"Warning: expected device '{device_name}' was not found in loaded data.")


def _warn_missing_expected_registers(dfs_dict: dict, device_registers: dict | None, verbose: bool) -> None:
    """Print warnings for expected registers that were not loaded."""
    if not verbose or not device_registers:
        return

    for device_name, expected_registers in device_registers.items():
        if device_name not in dfs_dict:
            print(f"Warning: expected device '{device_name}' was not found in loaded data.")
            continue

        loaded = dfs_dict[device_name]
        if not isinstance(loaded, dict):
            continue

        missing = [str(register) for register in expected_registers if str(register) not in loaded]
        for register in missing:
            print(f"Warning: expected register '{register}' was not found for device '{device_name}'.")


def _filter_to_matching_times(data_arrays: dict[str, xr.DataArray], verbose: bool) -> dict[str, xr.DataArray]:
    """
    Keep only timestamps present in every DataArray.

    If any DataArray lacks a ``Time`` coordinate, the input dict is returned
    unchanged.
    """
    if not data_arrays:
        return data_arrays

    time_sets = []
    for data_array in data_arrays.values():
        if "Time" not in data_array.coords:
            return data_arrays
        time_sets.append(set(data_array.get_index("Time")))

    common_times = sorted(set.intersection(*time_sets)) if time_sets else []
    filtered = {
        name: data_array.sel(Time=common_times)
        for name, data_array in data_arrays.items()
    }

    if verbose:
        print(f"Filtered to {len(common_times)} matching timestamps.")

    return filtered


################################################################################
# Device
################################################################################


class Device(DataSource):
    """
    Minimal device wrapper for HARP-style binary data.

    This class only adds the device-specific behavior that DataSource should
    not own directly:
    - use ``read_harp_bin`` for ``.bin`` files
    - select device folders by prefix
    - collapse duplicated device folder levels
    - rekey HARP stems to register keys
    - optionally warn about missing expected devices or registers
    - optionally build named device_data_arrays
    """

    def __init__(
        self,
        #== data input setup ===#
        dfs_dict: dict | None = None,
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = "./device.yml",
        device_type: str = "Behavior",
        #== expected device layout ===#
        device_list: list[str] | None = None,
        device_IDs: dict | None = None,
        device_registers: dict | None = None,
        #== named data outputs ===#
        device_data_arrays: dict | None = None,
        #== loader behavior ===#
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialise a device wrapper around DataSource.

        Parameters
        ----------
        dfs_dict : dict or None
            Pre-loaded nested data tree. If provided, directory loading is
            skipped.
        experiment_directory_path : str or Path or None
            Root path passed to DataSource when loading from disk.
        harp_device_yaml_path : str or Path
            YAML definition passed through to ``read_harp_bin``.
        device_type : str
            Prefix used to select relevant top-level device folders.
        device_list : list[str] or None
            Expected device names. Used only for warning messages.
        device_IDs : dict or None
            Placeholder metadata for expected device IDs.
        device_registers : dict or None
            Expected registers per device. Also used as an optional filter.
        device_data_arrays : dict or None
            Mapping of friendly names to ``(device, register)`` paths.
        verbose : bool
            If True, print warnings for missing expected content.
        **kwargs
            Additional selector kwargs passed through to DataSource.
        """
        self.harp_device_yaml_path = harp_device_yaml_path
        self.device_type = device_type
        self.device_list = device_list
        self.device_IDs = device_IDs
        self.device_registers = device_registers
        self.device_data_arrays = device_data_arrays

        super().__init__(
            dfs_dict=dfs_dict,
            experiment_directory_path=experiment_directory_path,
            readers={".bin": read_harp_bin},
            reader_kwargs={".bin": {"harp_device_yaml_path": harp_device_yaml_path}},
            datasource_data_arrays=None,
            verbose=verbose,
            l0_selector=starts_with(device_type),
            **kwargs,
        )

        self.dfs_dict = _collapse_device_folder_level(self.dfs_dict)
        self.dfs_dict = _rekey_harp_registers(self.dfs_dict)
        self.dfs_dict = _filter_device_registers(self.dfs_dict, device_registers)

        _warn_missing_expected_devices(self.dfs_dict, device_list, verbose)
        _warn_missing_expected_registers(self.dfs_dict, device_registers, verbose)

        self.data_arrays = {}
        if device_data_arrays is not None:
            self.data_arrays = _build_data_arrays(
                dfs_dict=self.dfs_dict,
                datasource_data_arrays=device_data_arrays,
                verbose=verbose,
            )


################################################################################
# Presets
################################################################################


class SoundCard(Device):
    """
    Preset device wrapper for SoundCard data.

    This preset provides default SoundCard devices, registers, and named
    device_data_arrays. It adds no extra helper methods beyond the base class.
    """

    def __init__(
        self,
        #== data input setup ===#
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = "./soundcard.yml",
        #== expected device layout ===#
        device_list: list[str] | None = ["SoundCard"],
        device_IDs: dict | None = {"SoundCard": "SoundCard0"},
        device_registers: dict | None = {"SoundCard": ["8", "32", "33", "35"]},
        #== named data outputs ===#
        device_data_arrays: dict | None = {
            "PlaySoundFreq": ("SoundCard", "32"),
            "StopLog": ("SoundCard", "33"),
            "AttenuationRight": ("SoundCard", "35"),
        },
        #== loader behavior ===#
        verbose: bool = False,
        **kwargs,
    ):
        """Initialise the SoundCard preset."""

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type="SoundCard",
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers=device_registers,
            device_data_arrays=device_data_arrays,
            verbose=verbose,
            **kwargs,
        )


class CameraStart(Device):
    """
    Preset device wrapper for camera start registers.

    This preset provides default behavior devices and exposes the start-camera
    registers as named device_data_arrays. It adds no extra helper methods.
    """

    def __init__(
        self,
        #== data input setup ===#
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = "./device.yml",
        #== expected device layout ===#
        device_list: list[str] | None = [
            "Behavior0",
            "Behavior1",
            "Behavior2",
            "Behavior3",
            "Behavior4",
            "Behavior5",
        ],
        device_IDs: dict | None = {
            "Behavior0": "ID_0",
            "Behavior1": "ID_1",
            "Behavior2": "ID_2",
            "Behavior3": "ID_3",
            "Behavior4": "ID_4",
            "Behavior5": "ID_5",
        },
        device_registers: dict | None = {
            "Behavior0": ["78"],
            "Behavior1": ["78"],
            "Behavior2": ["78"],
            "Behavior3": ["78"],
            "Behavior4": ["78"],
            "Behavior5": ["78"],
        },
        #== named data outputs ===#
        device_data_arrays: dict | None = {
            "Behavior0_StartCameras": ("Behavior0", "78"),
            "Behavior1_StartCameras": ("Behavior1", "78"),
            "Behavior2_StartCameras": ("Behavior2", "78"),
            "Behavior3_StartCameras": ("Behavior3", "78"),
            "Behavior4_StartCameras": ("Behavior4", "78"),
            "Behavior5_StartCameras": ("Behavior5", "78"),
        },
        #== loader behavior ===#
        verbose: bool = False,
        **kwargs,
    ):
        """Initialise the CameraStart preset."""

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type="Behavior",
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers=device_registers,
            device_data_arrays=device_data_arrays,
            verbose=verbose,
            **kwargs,
        )


class Camera0Frames(Device):
    """
    Preset device wrapper for camera frame registers.

    In addition to the base Device behavior, this preset can call the
    matching-time helper to keep only timestamps shared by every loaded
    DataArray.
    """

    def __init__(
        self,
        #== data input setup ===#
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = "./device.yml",
        #== expected device layout ===#
        device_list: list[str] | None = [
            "Behavior0",
            "Behavior1",
            "Behavior2",
            "Behavior3",
            "Behavior4",
            "Behavior5",
        ],
        device_IDs: dict | None = {
            "Behavior0": "ID_0",
            "Behavior1": "ID_1",
            "Behavior2": "ID_2",
            "Behavior3": "ID_3",
            "Behavior4": "ID_4",
            "Behavior5": "ID_5",
        },
        device_registers: dict | None = {
            "Behavior0": ["92"],
            "Behavior1": ["92"],
            "Behavior2": ["92"],
            "Behavior3": ["92"],
            "Behavior4": ["92"],
            "Behavior5": ["92"],
        },
        #== named data outputs ===#
        device_data_arrays: dict | None = {
            "Behavior0_Camera0Frame": ("Behavior0", "92"),
            "Behavior1_Camera0Frame": ("Behavior1", "92"),
            "Behavior2_Camera0Frame": ("Behavior2", "92"),
            "Behavior3_Camera0Frame": ("Behavior3", "92"),
            "Behavior4_Camera0Frame": ("Behavior4", "92"),
            "Behavior5_Camera0Frame": ("Behavior5", "92"),
        },
        #== timing helper mode ===#
        matching_only: bool = False,
        #== loader behavior ===#
        verbose: bool = False,
        **kwargs,
    ):
        """Initialise the Camera0Frames preset."""

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type="Behavior",
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers=device_registers,
            device_data_arrays=device_data_arrays,
            verbose=verbose,
            **kwargs,
        )

        if matching_only:
            self.data_arrays = _filter_to_matching_times(self.data_arrays, verbose)




__all__ = [
    "Device",
    "SoundCard",
    "CameraStart",
    "Camera0Frames",
]
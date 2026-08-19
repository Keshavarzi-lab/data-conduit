"""
Device wrappers built on top of MonoSource.
------------------------------------------

Description:
    This module provides a minimal device layer for HARP-style data.
    It keeps MonoSource generic and adds only the device-specific steps
    needed to load .bin files, reshape the resulting dfs_dict into a
    device/register layout, optionally validate expected content, and
    expose named DataArrays.

Contents:
    - Device
        Minimal HARP-style device wrapper around MonoSource.
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

from data_conduit.actual.datasources.monosource.monosource_core import MonoSource
from data_conduit.actual.integrations.harp.harptools import collect_harp_dfs

################################################################################



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


class Device(MonoSource):
    """
    Minimal device wrapper for HARP-style binary data.

    This class only adds the device-specific behavior that MonoSource should
    not own directly:
    - use ``read_harp_bin`` for ``.bin`` files
    - select device folders by prefix
    - collapse duplicated device folder levels
    - rekey HARP stems to register keys
    - optionally warn about missing expected devices or registers
    - optionally build named device_data_arrays
    
    Parameters
    ----------
    dfs_dict : dict or None
        Pre-loaded nested data tree. If provided, directory loading is
        skipped.
    experiment_directory_path : str or Path or None
        Root path passed to MonoSource when loading from disk.
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
    rekey : bool
        If True, rekey HARP stems to register keys (e.g. convert
        ``Behavior0_8_timestamp`` into ``8``) so the device tree becomes
        ``device -> register -> dataframe``. Set to False to keep original keys.
    verbose : bool
        If True, print warnings for missing expected content and info about loading steps.
    **kwargs
        Additional selector kwargs passed through to MonoSource.
    
    Attributes
    ----------
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from device_data_arrays.
        Accessed via device.data_arrays['{friendly_name}']
            
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
        rekey: bool = True,
        #== loader behavior ===#
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialise a device wrapper around MonoSource.

        Parameters
        ----------
        dfs_dict : dict or None
            Pre-loaded nested data tree. If provided, directory loading is
            skipped.
        experiment_directory_path : str or Path or None
            Root path passed to MonoSource when loading from disk.
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
        rekey : bool
            If True, rekey HARP stems to register keys (e.g. convert
            ``Behavior0_8_timestamp`` into ``8``) so the device tree becomes
            ``device -> register -> dataframe``. Set to False to keep original keys.
        verbose : bool
            If True, print warnings for missing expected content.
        **kwargs
            Additional selector kwargs passed through to MonoSource.
        """
        self.harp_device_yaml_path = harp_device_yaml_path
        self.device_type = device_type
        self.device_list = device_list
        self.device_IDs = device_IDs
        self.device_registers = device_registers
        self.device_data_arrays = device_data_arrays

        # Load via collect_harp_dfs if no dfs_dict provided
        if dfs_dict is None and experiment_directory_path is not None:
            dfs_dict = collect_harp_dfs(
                base_path=experiment_directory_path,
                harp_device_yaml_path=harp_device_yaml_path,
                device_type=device_type,
                device_registers=device_registers,
                device_list=device_list,
                rekey=rekey,
                verbose=verbose,
                **kwargs,
            )

        super().__init__(
            dfs_dict=dfs_dict,
            experiment_directory_path=experiment_directory_path,
            monosource_data_arrays=device_data_arrays,
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

    Parameters
    ----------
    experiment_directory_path : str or Path or None
        Root path passed to MonoSource when loading from disk.
    harp_device_yaml_path : str or Path
        YAML definition passed through to ``read_harp_bin``.
    device_list : list[str] or None
        Expected device names. Used only for warning messages.
        SoundCard Default: ["SoundCard"]
    device_IDs : dict or None
        Placeholder metadata for expected device IDs.
        SoundCard Default: {"SoundCard": "SoundCard0"}

    device_registers : dict or None
        Expected registers per device. Also used as an optional filter.
        SoundCard Default: {"SoundCard": ["8", "32", "33", "35"]}
    device_data_arrays : dict or None
        Mapping of friendly names to ``(device, register)`` paths.
        SoundCard Default: {
                "PlaySoundFreq": {"l0_selector": "SoundCard", "l1_selector": "32"},
                "StopLog": {"l0_selector": "SoundCard", "l1_selector": "33"},
                "AttenuationRight": {"l0_selector": "SoundCard", "l1_selector": "35"},
            }
    rekey : bool
        If True, rekey HARP stems to register keys (e.g. convert
        ``SoundCard_8_1904-01-01T01-00-00.bin`` into ``32``) so the device tree becomes 
        ``device -> register -> dataframe``. 
        Set to False to keep original keys.
    verbose : bool
        If True, print verbose messages during initialisation.
    **kwargs
        Additional selector kwargs passed through to MonoSource.
    
    Attributes
    ----------
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from device_data_arrays.
        Accessed via device.data_arrays['{friendly_name}']

        Default SoundCard data_arrays:
        {
            "PlaySoundFreq": xr.DataArray for SoundCard register 32,
            "StopLog": xr.DataArray for SoundCard register 33,
            "AttenuationRight": xr.DataArray for SoundCard register 35,
        }
    
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
                "PlaySoundFreq": {"l0_selector": "SoundCard", "l1_selector": "32"},
                "StopLog": {"l0_selector": "SoundCard", "l1_selector": "33"},
                "AttenuationRight": {"l0_selector": "SoundCard", "l1_selector": "35"},
            },
        rekey: bool = True,
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
            rekey=rekey,
            verbose=verbose,
            **kwargs,
        )


class CameraStart(Device):
    """
    Preset device wrapper for camera start registers.

    This preset provides default behavior devices and exposes the start-camera
    registers as named device_data_arrays. It adds no extra helper methods.

    Parameters
    ----------
    experiment_directory_path : str or Path or None
        Root path passed to MonoSource when loading from disk.
    harp_device_yaml_path : str or Path
        YAML definition passed through to ``read_harp_bin``.
    device_list : list[str] or None
        Expected device names. Used only for warning messages.
        CameraStart Default: ["Behavior0", "Behavior1", "Behavior2", "Behavior3", "Behavior4", "Behavior5"]
    device_IDs : dict or None
        Placeholder metadata for expected device IDs.
        CameraStart Default: {
            "Behavior0": "ID_0",
            "Behavior1": "ID_1",
            "Behavior2": "ID_2",
            "Behavior3": "ID_3",
            "Behavior4": "ID_4",
            "Behavior5": "ID_5",
        }
    device_registers : dict or None
        Expected registers per device. Also used as an optional filter.
        CameraStart Default: {
            "Behavior0": ["78"],
            "Behavior1": ["78"],
            "Behavior2": ["78"],
            "Behavior3": ["78"],
            "Behavior4": ["78"],
            "Behavior5": ["78"],
        }
    device_data_arrays : dict or None
        Mapping of friendly names to ``(device, register)`` paths.
        CameraStart Default: {
            "Behavior0_StartCameras": {"l0_selector": "Behavior0", "l1_selector": "78"},
            "Behavior1_StartCameras": {"l0_selector": "Behavior1", "l1_selector": "78"},
            "Behavior2_StartCameras": {"l0_selector": "Behavior2", "l1_selector": "78"},
            "Behavior3_StartCameras": {"l0_selector": "Behavior3", "l1_selector": "78"},
            "Behavior4_StartCameras": {"l0_selector": "Behavior4", "l1_selector": "78"},
            "Behavior5_StartCameras": {"l0_selector": "Behavior5", "l1_selector": "78"},
        }
    rekey : bool
        If True, rekey HARP stems to register keys (e.g. convert
        ``Behavior0_78_1904-01-01T01-00-00.bin`` into ``78``) so the device tree becomes 
        ``device -> register -> dataframe``. 
        Set to False to keep original keys.
    verbose : bool
        If True, print verbose messages during initialisation.
    **kwargs
        Additional selector kwargs passed through to MonoSource.

    Attributes
    ----------
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from device_data_arrays.
        Accessed via device.data_arrays['{friendly_name}']
        Default CameraStart data_arrays:
        {
            "Behavior0_StartCameras": xr.DataArray for Behavior0 register 78,
            "Behavior1_StartCameras": xr.DataArray for Behavior1 register 78,
            "Behavior2_StartCameras": xr.DataArray for Behavior2 register 78,
            "Behavior3_StartCameras": xr.DataArray for Behavior3 register 78,
            "Behavior4_StartCameras": xr.DataArray for Behavior4 register 78,
            "Behavior5_StartCameras": xr.DataArray for Behavior5 register 78,
        }
    """

    def __init__(
        self,
        #== data input setup ===#
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = "./device.yml",
        #== expected device layout ===#
        device_list: list[str] | None = ["Behavior0", "Behavior1", "Behavior2", "Behavior3", "Behavior4", "Behavior5"],
        device_IDs: dict | None = {"Behavior0": "ID_0","Behavior1": "ID_1","Behavior2": "ID_2","Behavior3": "ID_3","Behavior4": "ID_4","Behavior5": "ID_5",},
        device_registers: dict | None = {"Behavior0": ["78"],"Behavior1": ["78"],"Behavior2": ["78"],"Behavior3": ["78"],"Behavior4": ["78"],"Behavior5": ["78"]},
        #== named data outputs ===#
        device_data_arrays: dict | None = {
            "Behavior0_StartCameras": {"l0_selector": "Behavior0", "l1_selector": "78"},
            "Behavior1_StartCameras": {"l0_selector": "Behavior1", "l1_selector": "78"},
            "Behavior2_StartCameras": {"l0_selector": "Behavior2", "l1_selector": "78"},
            "Behavior3_StartCameras": {"l0_selector": "Behavior3", "l1_selector": "78"},
            "Behavior4_StartCameras": {"l0_selector": "Behavior4", "l1_selector": "78"},
            "Behavior5_StartCameras": {"l0_selector": "Behavior5", "l1_selector": "78"},
        },
        rekey: bool = True,
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
            rekey= rekey,
            verbose=verbose,
            **kwargs,
        )


class Camera0Frames(Device):
    """
    Preset device wrapper for camera frame registers.

    In addition to the base Device behavior, this preset can call the
    matching-time helper to keep only timestamps shared by every loaded
    DataArray.
    
    Parameters
    ----------
    experiment_directory_path : str or Path or None
        Root path passed to MonoSource when loading from disk.
    harp_device_yaml_path : str or Path
        YAML definition passed through to ``read_harp_bin``.
    device_list : list[str] or None
        Expected device names. Used only for warning messages.
        Camera0Frames Default: ["Behavior0", "Behavior1", "Behavior2", "Behavior3", "Behavior4", "Behavior5"]
    device_IDs : dict or None
        Placeholder metadata for expected device IDs.
        Camera0Frames Default: {
            "Behavior0": "ID_0",
            "Behavior1": "ID_1",
            "Behavior2": "ID_2",
            "Behavior3": "ID_3",
            "Behavior4": "ID_4",
            "Behavior5": "ID_5",
        }
    device_registers : dict or None
        Expected registers per device. Also used as an optional filter.
        Camera0Frames Default: {
            "Behavior0": ["92"],
            "Behavior1": ["92"],
            "Behavior2": ["92"],
            "Behavior3": ["92"],
            "Behavior4": ["92"],
            "Behavior5": ["92"],
        }
    device_data_arrays : dict or None
        Mapping of friendly names to ``(device, register)`` paths.
        Camera0Frames Default: {
            "Behavior0_Camera0Frame": {"l0_selector": "Behavior0", "l1_selector": "92"},
            "Behavior1_Camera0Frame": {"l0_selector": "Behavior1", "l1_selector": "92"},
            "Behavior2_Camera0Frame": {"l0_selector": "Behavior2", "l1_selector": "92"},
            "Behavior3_Camera0Frame": {"l0_selector": "Behavior3", "l1_selector": "92"},
            "Behavior4_Camera0Frame": {"l0_selector": "Behavior4", "l1_selector": "92"},
            "Behavior5_Camera0Frame": {"l0_selector": "Behavior5", "l1_selector": "92"},
        }
    rekey : bool
        If True, rekey HARP stems to register keys (e.g. convert
        ``Behavior0_92_1904-01-01T01-00-00.bin`` into ``92``) so the device tree becomes 
        ``device -> register -> dataframe``. 
        Set to False to keep original keys.
    matching_only : bool
        If True, apply the matching-time helper to keep only timestamps shared by every loaded DataArray.
    verbose : bool
        If True, print verbose messages during initialisation.
    **kwargs
        Additional selector kwargs passed through to MonoSource.

    Attributes
    ----------
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from device_data_arrays.
        Accessed via device.data_arrays['{friendly_name}']
        Default Camera0Frames data_arrays:
        {
            "Behavior0_Camera0Frame": xr.DataArray for Behavior0 register 92,
            "Behavior1_Camera0Frame": xr.DataArray for Behavior1 register 92,
            "Behavior2_Camera0Frame": xr.DataArray for Behavior2 register 92,
            "Behavior3_Camera0Frame": xr.DataArray for Behavior3 register 92,
            "Behavior4_Camera0Frame": xr.DataArray for Behavior4 register 92,
            "Behavior5_Camera0Frame": xr.DataArray for Behavior5 register 92,
        }
    """

    def __init__(
        self,
        #== data input setup ===#
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = "./device.yml",
        #== expected device layout ===#
        device_list: list[str] | None = ["Behavior0", "Behavior1", "Behavior2", "Behavior3", "Behavior4", "Behavior5"],
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
            "Behavior0_Camera0Frame": {"l0_selector": "Behavior0", "l1_selector": "92"},
            "Behavior1_Camera0Frame": {"l0_selector": "Behavior1", "l1_selector": "92"},
            "Behavior2_Camera0Frame": {"l0_selector": "Behavior2", "l1_selector": "92"},
            "Behavior3_Camera0Frame": {"l0_selector": "Behavior3", "l1_selector": "92"},
            "Behavior4_Camera0Frame": {"l0_selector": "Behavior4", "l1_selector": "92"},
            "Behavior5_Camera0Frame": {"l0_selector": "Behavior5", "l1_selector": "92"},
        },
        rekey: bool = True,
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
            rekey= rekey,
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
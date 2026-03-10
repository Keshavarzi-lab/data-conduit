'''
MultiDevice and preset subclasses.
----------------------------------

Description:
    MultiDevice extends MultiSource for HARP-style multi-device data.
    It mirrors the role of Device in datasource land: resolve/load a
    device-oriented dfs_dict, then hand off to the base class for output
    construction. The key difference is that outputs are virtual-map aligned
    multisource arrays rather than direct named DataArray paths.

Contents:
----------------------------------
- MultiDevice(MultiSource)
    Thin device-oriented wrapper around MultiSource.
- Nosepoke(MultiDevice)
    Preset for common behavior nosepoke configuration.
'''

################################################################################
# Imports
################################################################################

from pathlib import Path

from data_conduit.datasource.datasource_core import _obtain_dfs_dict
from data_conduit.harptools import read_harp_bin
from data_conduit.multisource.multisource_core import MultiSource
from data_conduit.utils import starts_with

################################################################################


def _collapse_device_folder_level(dfs_dict: dict) -> dict:
    '''Collapse duplicated device folder levels in loaded HARP trees.'''
    collapsed = {}
    for device_name, value in dfs_dict.items():
        if isinstance(value, dict) and list(value.keys()) == [device_name]:
            collapsed[device_name] = value[device_name]
        else:
            collapsed[device_name] = value
    return collapsed


def _rekey_harp_registers(dfs_dict: dict) -> dict:
    '''Rekey HARP file stems (e.g. Behavior0_32_timestamp) to register keys.'''
    rekeyed = {}
    for device_name, value in dfs_dict.items():
        if not isinstance(value, dict):
            rekeyed[device_name] = value
            continue

        register_dict = {}
        for key, item in value.items():
            parts = key.split('_')
            register_key = parts[1] if len(parts) > 1 else key
            if register_key in register_dict:
                raise ValueError(
                    f"Rekeying would overwrite data for device '{device_name}' at "
                    f"register '{register_key}'."
                )
            register_dict[register_key] = item

        rekeyed[device_name] = register_dict

    return rekeyed


def _filter_device_registers(dfs_dict: dict, device_registers: dict | None) -> dict:
    '''Keep only requested registers for devices listed in device_registers.'''
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


def _warn_missing_expected_devices(
    dfs_dict: dict,
    device_list: list[str] | None,
    verbose: bool,
) -> None:
    '''Warn for expected devices that are missing from loaded data.'''
    if not verbose or not device_list:
        return

    missing = [device_name for device_name in device_list if device_name not in dfs_dict]
    for device_name in missing:
        print(f"Warning: expected device '{device_name}' was not found in loaded data.")


def _warn_missing_expected_registers(
    dfs_dict: dict,
    device_registers: dict | None,
    verbose: bool,
) -> None:
    '''Warn for expected registers that are missing from loaded data.'''
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


def _obtain_multidevice_dfs_dict(
    dfs_dict: dict | None,
    experiment_directory_path: str | Path | None,
    harp_device_yaml_path: str | Path,
    device_type: str,
    device_list: list[str] | None,
    device_IDs: dict | None,
    device_registers: dict | None,
    verbose: bool,
    **kwargs,
) -> dict:
    '''Resolve dfs_dict for MultiDevice without depending on Device class internals.'''
    if dfs_dict is not None:
        if not isinstance(dfs_dict, dict):
            raise TypeError(
                f"dfs_dict must be a dict if provided, got {type(dfs_dict).__name__}."
            )
        return dfs_dict

    if experiment_directory_path is None:
        raise ValueError(
            'Provide either dfs_dict or experiment_directory_path for MultiDevice.'
        )

    loaded = _obtain_dfs_dict(
        dfs_dict=None,
        experiment_directory_path=experiment_directory_path,
        readers={'.bin': read_harp_bin},
        reader_kwargs={'.bin': {'harp_device_yaml_path': harp_device_yaml_path}},
        keep_empty=False,
        flatten=False,
        separator=':',
        verbose=verbose,
        l0_selector=starts_with(device_type),
        **kwargs,
    )

    loaded = _collapse_device_folder_level(loaded)
    loaded = _rekey_harp_registers(loaded)
    loaded = _filter_device_registers(loaded, device_registers)

    _warn_missing_expected_devices(loaded, device_list, verbose)
    _warn_missing_expected_registers(loaded, device_registers, verbose)

    return loaded


def _build_virtual_maps(
    channel_list: list[str],
    data_registerIDs: dict[str, dict[str, str]],
    device_list: list[str],
    channel_localIDs: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    '''Build dict-style virtual maps from channel/device/register inputs.'''
    if not channel_list:
        raise ValueError('channel_list must be a non-empty list.')
    if not data_registerIDs:
        raise ValueError('data_registerIDs must be provided and non-empty.')
    if not device_list:
        raise ValueError('device_list must be provided when building virtual maps.')

    virtual_maps: dict[str, dict[str, dict[str, str]]] = {}

    for data_key, register_map in data_registerIDs.items():
        if not register_map:
            raise ValueError(f"data_registerIDs['{data_key}'] is empty.")

        local_ids = (
            channel_localIDs[data_key]
            if channel_localIDs is not None and data_key in channel_localIDs
            else list(register_map.keys())
        )

        n_local = len(local_ids)
        if n_local == 0:
            raise ValueError(f"No local_ids for data_key '{data_key}'.")

        if len(channel_list) % n_local != 0:
            raise ValueError(
                f'len(channel_list)={len(channel_list)} is not a multiple of '
                f"len(localIDs)={n_local} for data_key '{data_key}'."
            )

        n_required_devices = len(channel_list) // n_local
        if n_required_devices > len(device_list):
            raise ValueError(
                f"data_key '{data_key}' requires {n_required_devices} devices "
                f'from channel_list/localIDs, but only {len(device_list)} were provided.'
            )

        vmap: dict[str, dict[str, str]] = {}
        for idx, channel in enumerate(channel_list):
            dev_idx = idx // n_local
            local_id = local_ids[idx % n_local]

            if local_id not in register_map:
                raise KeyError(
                    f"localID '{local_id}' missing from data_registerIDs['{data_key}']."
                )

            vmap[channel] = {
                'device': device_list[dev_idx],
                'register': str(register_map[local_id]),
                'localID': str(local_id),
            }

        virtual_maps[data_key] = vmap

    return virtual_maps


################################################################################
# MultiDevice
################################################################################


class MultiDevice(MultiSource):
    '''
    Device-oriented MultiSource wrapper.

    Like Device, this class focuses on obtaining the right dfs_dict shape.
    Unlike Device, it then delegates to MultiSource to construct virtual-map
    aligned outputs (data_arrays, lookup_arrays, lookup_virtual_coords).
    '''

    def __init__(
        self,
        #== data input setup ==#
        dfs_dict: dict | None = None,
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = './device.yml',
        device_type: str = 'Behavior',
        #== expected device layout ==#
        device_list: list[str] | None = None,
        device_IDs: dict | None = None,
        device_registers: dict | None = None,
        #== virtual map setup ==#
        virtual_maps: dict | None = None,
        data_keys: list[str] | None = None,
        data_registerIDs: dict[str, dict[str, str]] | None = None,
        channel_list: list[str] | None = None,
        channel_localIDs: dict[str, list[str]] | None = None,
        #== multisource config ==#
        global_coord_name: str = 'global_coord',
        virtual_coord_names: list[str] | None = None,
        dict_of: str = 'dicts',
        data_array_names: dict | None = None,
        data_array_attrs: dict | None = None,
        lookup_array_names: dict | None = None,
        lookup_array_attrs: dict | None = None,
        test_values: bool = False,
        fill_value=None,
        verbose: bool = False,
        **kwargs,
    ):
        '''Initialise MultiDevice and optionally load/build maps before MultiSource.'''
        self.experiment_directory_path = experiment_directory_path
        self.harp_device_yaml_path = harp_device_yaml_path
        self.device_type = device_type
        self.device_list = device_list
        self.device_IDs = device_IDs
        self.device_registers = device_registers

        resolved_dfs_dict = _obtain_multidevice_dfs_dict(
            dfs_dict=dfs_dict,
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers=device_registers,
            verbose=verbose,
            **kwargs,
        )

        resolved_virtual_maps = virtual_maps
        if (
            resolved_virtual_maps is None
            and data_registerIDs is not None
            and channel_list is not None
            and device_list is not None
        ):
            resolved_virtual_maps = _build_virtual_maps(
                channel_list=channel_list,
                data_registerIDs=data_registerIDs,
                device_list=device_list,
                channel_localIDs=channel_localIDs,
            )

        if virtual_coord_names is None and dict_of == 'dicts':
            virtual_coord_names = ['device', 'register', 'localID']

        super().__init__(
            dfs_dict=resolved_dfs_dict,
            virtual_maps=resolved_virtual_maps,
            data_keys=data_keys,
            global_coord_name=global_coord_name,
            virtual_coord_names=virtual_coord_names,
            dict_of=dict_of,
            data_array_names=data_array_names,
            data_array_attrs=data_array_attrs,
            lookup_array_names=lookup_array_names,
            lookup_array_attrs=lookup_array_attrs,
            test_values=test_values,
            fill_value=fill_value,
            verbose=verbose,
        )


################################################################################
# Nosepoke Preset
################################################################################


class Nosepoke(MultiDevice):
    '''Preset MultiDevice for common nosepoke peripheral data configuration.'''

    def __init__(
        self,
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = './device.yml',
        device_type: str = 'Behavior',
        device_list: list[str] | None = None,
        channel_list: list[str] | None = None,
        data_registerIDs: dict[str, dict[str, str]] | None = None,
        data_keys: list[str] | None = None,
        global_coord_name: str = 'peripherals',
        virtual_coord_names: list[str] | None = None,
        dict_of: str = 'dicts',
        data_array_names: dict | None = None,
        data_array_attrs: dict | None = None,
        lookup_array_names: dict | None = None,
        lookup_array_attrs: dict | None = None,
        test_values: bool = False,
        fill_value=None,
        verbose: bool = False,
        **kwargs,
    ):
        '''Initialise Nosepoke preset and delegate to MultiDevice.'''
        if device_list is None:
            device_list = [f'Behavior{i}' for i in range(6)]

        if channel_list is None:
            channel_list = [f'NP_{i}' for i in range(18)]

        if data_registerIDs is None:
            data_registerIDs = {
                'Activations': {
                    'DIPort0': '32',
                    'DIPort1': '32',
                    'DIPort2': '32',
                },
                'LEDs': {
                    'DOPort0': '34',
                    'DOPort1': '34',
                    'DOPort2': '34',
                },
                'Valves': {
                    'DOPort3': '34',
                    'DOPort4': '34',
                    'DOPort5': '34',
                },
                'Rewards': {
                    'DIPort3': '32',
                    'DIPort4': '32',
                    'DIPort5': '32',
                },
            }

        if data_keys is None:
            data_keys = list(data_registerIDs.keys())

        device_registers = {
            device_name: sorted(
                {
                    str(register)
                    for key in data_keys
                    for register in data_registerIDs[key].values()
                }
            )
            for device_name in device_list
        }

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_registers=device_registers,
            virtual_maps=None,
            data_keys=data_keys,
            data_registerIDs={key: data_registerIDs[key] for key in data_keys},
            channel_list=channel_list,
            global_coord_name=global_coord_name,
            virtual_coord_names=virtual_coord_names,
            dict_of=dict_of,
            data_array_names=data_array_names,
            data_array_attrs=data_array_attrs,
            lookup_array_names=lookup_array_names,
            lookup_array_attrs=lookup_array_attrs,
            test_values=test_values,
            fill_value=fill_value,
            verbose=verbose,
            **kwargs,
        )

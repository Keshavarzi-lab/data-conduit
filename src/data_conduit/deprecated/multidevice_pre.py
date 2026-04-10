'''
MultiDevice module.
-------------------

Description:
    Device-specific MultiSource orchestration for HARP-style nested data.
    This module bridges DataSource/HarpDevice loading and MultiSource virtual
    array construction.

Contents:
--------------------------------
- MultiDevice
    High-level class that:
      1) Loads HARP device DataFrames via HarpDevice (or accepts pre-loaded dfs_dict)
      2) Builds/uses virtual maps
      3) Constructs data arrays + lookup arrays via MultiSource
- Nosepoke
    Preset wrapper for common behavior nosepoke workflows.
'''


################################################################################
# Imports
################################################################################

from pathlib import Path

from data_conduit.datasources.monosource.devices import HarpDevice
from data_conduit.datasources.multisource.multisource_core import MultiSource

################################################################################




################################################################################
# MultiDevice
################################################################################

class MultiDevice(MultiSource):
    '''
    HARP-specific MultiSource orchestrator.

    This class composes:
    - `HarpDevice` for loading nested register DataFrames, and
    - `MultiSource` for creating virtual DataArrays/lookup arrays.
    '''

    def __init__(
        self,
        # Loading inputs
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = './device.yml',
        device_type: str = 'Behavior',
        device_list: list[str] | None = None,
        device_IDs: dict | None = None,
        device_registers_dict: dict | None = None,
        dfs_dict: dict | None = None,
        # Virtual-map inputs
        virtual_maps: dict | None = None,
        data_keys: list[str] | None = None,
        data_registerIDs: dict[str, dict[str, str]] | None = None,
        channel_list: list[str] | None = None,
        channel_localIDs: dict[str, list[str]] | None = None,
        # MultiSource config
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
        '''Initialise MultiDevice and optionally auto-load + auto-build maps.'''
        self.experiment_directory_path = experiment_directory_path
        self.harp_device_yaml_path = harp_device_yaml_path
        self.device_type = device_type
        self.device_list = device_list
        self.device_IDs = device_IDs
        self.device_registers_dict = device_registers_dict
        self.verbose = verbose

        #=== 1) Resolve dfs_dict
        if dfs_dict is None and experiment_directory_path is not None:
            loader = HarpDevice(
                experiment_directory_path=experiment_directory_path,
                harp_device_yaml_path=harp_device_yaml_path,
                device_type=device_type,
                device_list=device_list,
                device_IDs=device_IDs,
                device_registers_dict=device_registers_dict,
                verbose=verbose,
                **kwargs,
            )
            dfs_dict = loader.dfs_dict

        #=== 2) Resolve virtual maps
        if virtual_maps is None and data_registerIDs is not None and channel_list is not None:
            virtual_maps = self.build_virtual_maps(
                channel_list=channel_list,
                data_registerIDs=data_registerIDs,
                device_list=device_list,
                channel_localIDs=channel_localIDs,
            )

        #=== 3) Default virtual coord names for dicts-form maps
        if virtual_coord_names is None:
            virtual_coord_names = ['device', 'register', 'localID']

        super().__init__(
            dfs_dict=dfs_dict if dfs_dict is not None else {},
            virtual_maps=virtual_maps if virtual_maps is not None else {},
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

    @staticmethod
    def build_virtual_maps(
        channel_list: list[str],
        data_registerIDs: dict[str, dict[str, str]],
        device_list: list[str] | None,
        channel_localIDs: dict[str, list[str]] | None = None,
    ) -> dict[str, dict[str, dict[str, str]]]:
        '''
        Build dict-of-dicts virtual maps from a channel layout.

        Channel assignment uses device-major ordering:
            channel_list[i] -> device_list[i // n_local], localID[i % n_local]
        '''
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
                raise ValueError(f"No localIDs for data_key '{data_key}'.")

            if len(channel_list) % n_local != 0:
                raise ValueError(
                    f'len(channel_list)={len(channel_list)} is not a multiple of '
                    f"len(local_ids)={n_local} for data_key '{data_key}'."
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
                    'localID': local_id,
                }

            virtual_maps[data_key] = vmap

        return virtual_maps

    @staticmethod
    def infer_data_registerIDs(
        virtual_maps: dict[str, dict[str, dict[str, str]]],
    ) -> dict[str, dict[str, str]]:
        '''Infer `{data_key: {localID: register}}` from dict-style virtual maps.'''
        inferred: dict[str, dict[str, str]] = {}
        for data_key, vmap in virtual_maps.items():
            inferred[data_key] = {}
            for entry in vmap.values():
                if not isinstance(entry, dict):
                    continue
                local_id = entry.get('localID')
                register = entry.get('register')
                if local_id is None or register is None:
                    continue
                inferred[data_key].setdefault(str(local_id), str(register))
        return inferred

    @property
    def channel_data_arrays(self):
        '''Alias for `data_arrays` keyed by data_key.'''
        return self.data_arrays

    @property
    def channel_lookup_arrays(self):
        '''Alias for `lookup_arrays` keyed by data_key.'''
        return self.lookup_arrays


################################################################################




################################################################################
# Nosepoke Preset
################################################################################

class Nosepoke(MultiDevice):
    '''
    Convenience preset for common behavior nosepoke pipelines.

    This preset keeps architecture aligned with the new `MultiSource` stack,
    while mirroring common defaults used in older workflows.
    '''

    def __init__(
        self,
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = './device.yml',
        device_type: str = 'Behavior',
        device_list: list[str] | None = None,
        # Virtual-map construction inputs
        channel_list: list[str] | None = None,
        data_registerIDs: dict[str, dict[str, str]] | None = None,
        data_keys: list[str] | None = None,
        # MultiSource config
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
        '''Initialise Nosepoke preset and delegate to `MultiDevice`.'''
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

        device_registers_dict = {
            dev: sorted({str(v) for dk in data_keys for v in data_registerIDs[dk].values()})
            for dev in device_list
        }

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_registers_dict=device_registers_dict,
            virtual_maps=None,
            data_keys=data_keys,
            data_registerIDs={k: data_registerIDs[k] for k in data_keys},
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


################################################################################
"""
Legacy duplicate implementation kept for reference only.
The active implementation ends above.

'''
MultiDevice module.
-------------------

Description:
    Device-specific MultiSource orchestration for HARP-style nested data.
    This module bridges DataSource/HarpDevice loading and MultiSource virtual
    array construction.

Contents:
--------------------------------
- MultiDevice
    High-level class that:
      1) Loads HARP device DataFrames via HarpDevice (or accepts pre-loaded dfs_dict)
      2) Builds/uses virtual maps
      3) Constructs data arrays + lookup arrays via MultiSource
- Nosepoke
    Preset wrapper for common behavior nosepoke workflows.
'''


################################################################################
# Imports
################################################################################

from pathlib import Path

from data_conduit.datasource.devices import HarpDevice
from data_conduit.multisource.multisource_core import MultiSource

################################################################################




################################################################################
# MultiDevice
################################################################################

class MultiDevice(MultiSource):
    '''
    HARP-specific MultiSource orchestrator.

    This class composes:
    - `HarpDevice` for loading nested register DataFrames, and
    - `MultiSource` for creating virtual DataArrays/lookup arrays.

    Parameters
    ----------
    experiment_directory_path : str | Path | None
        Root experiment directory for loading HARP data.
    harp_device_yaml_path : str | Path
        YAML used by the HARP reader.
    device_type : str
        Device prefix filter (e.g. 'Behavior', 'SoundCard').
    device_list : list[str] | None
        Ordered list of device names.
    device_IDs : dict | None
        Optional device ID mapping.
    device_registers_dict : dict | None
        Optional per-device register filtering.
    dfs_dict : dict | None
        Optional pre-loaded nested dictionary; if provided, no load is run.

    virtual_maps : dict[str, dict] | None
        Mapping of data_key -> virtual map.
    data_keys : list[str] | None
        Data keys to process. Defaults to virtual_maps keys.
    data_registerIDs : dict[str, dict[str, str]] | None
        Fallback map for building virtual_maps when virtual_maps is not provided.
        Shape: {data_key: {localID: register_address}}.
    channel_list : list[str] | None
        Ordered global coordinate names used to build virtual maps.
    channel_localIDs : dict[str, list[str]] | None
        Optional explicit localID ordering per data_key. If omitted, localIDs
        are inferred from data_registerIDs keys.

    global_coord_name : str
        Name of global coordinate dimension.
    virtual_coord_names : list[str] | None
        Defaults to ['device', 'register', 'localID'].
    dict_of : str
        'dicts' or 'tuples'.
    data_array_names : dict | None
        Optional output DataArray names.
    data_array_attrs : dict | None
        Optional output DataArray attrs.
    lookup_array_names : dict | None
        Optional lookup DataArray names.
    lookup_array_attrs : dict | None
        Optional lookup DataArray attrs.
    test_values : bool
        If True, construct arrays with test string values.
    fill_value : any
        Fill value for missing updates.
    verbose : bool
        Verbose output.
    **kwargs : dict
        Additional selector kwargs passed to HarpDevice load path.
    '''

    def __init__(
        self,
        # Loading inputs
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = './device.yml',
        device_type: str = 'Behavior',
        device_list: list[str] | None = None,
        device_IDs: dict | None = None,
        device_registers_dict: dict | None = None,
        dfs_dict: dict | None = None,
        # Virtual-map inputs
        virtual_maps: dict | None = None,
        data_keys: list[str] | None = None,
        data_registerIDs: dict[str, dict[str, str]] | None = None,
        channel_list: list[str] | None = None,
        channel_localIDs: dict[str, list[str]] | None = None,
        # MultiSource config
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
        self.experiment_directory_path = experiment_directory_path
        self.harp_device_yaml_path = harp_device_yaml_path
        self.device_type = device_type
        self.device_list = device_list
        self.device_IDs = device_IDs
        self.device_registers_dict = device_registers_dict
        self.verbose = verbose

        #=== 1) Resolve dfs_dict
        if dfs_dict is None and experiment_directory_path is not None:
            loader = HarpDevice(
                experiment_directory_path=experiment_directory_path,
                harp_device_yaml_path=harp_device_yaml_path,
                device_type=device_type,
                device_list=device_list,
                device_IDs=device_IDs,
                device_registers_dict=device_registers_dict,
                verbose=verbose,
                **kwargs,
            )
            dfs_dict = loader.dfs_dict

        #=== 2) Resolve virtual maps
        if virtual_maps is None and data_registerIDs is not None and channel_list is not None:
            virtual_maps = self.build_virtual_maps(
                channel_list=channel_list,
                data_registerIDs=data_registerIDs,
                device_list=device_list,
                channel_localIDs=channel_localIDs,
            )

        #=== 3) Default virtual coord names for dicts-form maps
        if virtual_coord_names is None:
            virtual_coord_names = ['device', 'register', 'localID']

        super().__init__(
            dfs_dict=dfs_dict if dfs_dict is not None else {},
            virtual_maps=virtual_maps if virtual_maps is not None else {},
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

    #===========================================================================
    # Helpers
    #===========================================================================

    @staticmethod
    def build_virtual_maps(
        channel_list: list[str],
        data_registerIDs: dict[str, dict[str, str]],
        device_list: list[str] | None,
        channel_localIDs: dict[str, list[str]] | None = None,
    ) -> dict[str, dict[str, dict[str, str]]]:
        '''
        Build dict-of-dicts virtual maps from a channel layout.

        Channel assignment uses device-major ordering:
            channel_list[i] -> device_list[i // n_local], localID[i % n_local]
        '''
        if not channel_list:
            raise ValueError("channel_list must be a non-empty list.")
        if not data_registerIDs:
            raise ValueError("data_registerIDs must be provided and non-empty.")
        if not device_list:
            raise ValueError("device_list must be provided when building virtual maps.")

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
                raise ValueError(f"No localIDs for data_key '{data_key}'.")

            if len(channel_list) % n_local != 0:
                raise ValueError(
                    f"len(channel_list)={len(channel_list)} is not a multiple of "
                    f"len(local_ids)={n_local} for data_key '{data_key}'."
                )

            n_required_devices = len(channel_list) // n_local
            if n_required_devices > len(device_list):
                raise ValueError(
                    f"data_key '{data_key}' requires {n_required_devices} devices "
                    f"from channel_list/localIDs, but only {len(device_list)} were provided."
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
                    'localID': local_id,
                }

            virtual_maps[data_key] = vmap

        return virtual_maps

    @staticmethod
    def infer_data_registerIDs(
        virtual_maps: dict[str, dict[str, dict[str, str]]],
    ) -> dict[str, dict[str, str]]:
        '''
        Infer `{data_key: {localID: register}}` from dict-style virtual maps.

        If multiple registers are present for the same localID within one data_key,
        the first seen mapping is retained.
        '''
        inferred: dict[str, dict[str, str]] = {}
        for data_key, vmap in virtual_maps.items():
            inferred[data_key] = {}
            for entry in vmap.values():
                if not isinstance(entry, dict):
                    continue
                local_id = entry.get('localID')
                register = entry.get('register')
                if local_id is None or register is None:
                    continue
                inferred[data_key].setdefault(str(local_id), str(register))
        return inferred

    #===========================================================================
    # Backwards-compatible aliases
    #===========================================================================

    @property
    def channel_data_arrays(self):
        '''Alias for `data_arrays` keyed by data_key.'''
        return self.data_arrays

    @property
    def channel_lookup_arrays(self):
        '''Alias for `lookup_arrays` keyed by data_key.'''
        return self.lookup_arrays


################################################################################




################################################################################
# Nosepoke Preset
################################################################################

class Nosepoke(MultiDevice):
    '''
    Convenience preset for common behavior nosepoke pipelines.

    This preset keeps architecture aligned with the new `MultiSource` stack,
    while mirroring common defaults used in older workflows.
    '''

    def __init__(
        self,
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = './device.yml',
        device_type: str = 'Behavior',
        device_list: list[str] | None = None,
        # Virtual-map construction inputs
        channel_list: list[str] | None = None,
        data_registerIDs: dict[str, dict[str, str]] | None = None,
        data_keys: list[str] | None = None,
        # MultiSource config
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

        # Optional load-time register filtering from data_registerIDs.
        device_registers_dict = {
            dev: sorted({str(v) for dk in data_keys for v in data_registerIDs[dk].values()})
            for dev in device_list
        }

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_registers_dict=device_registers_dict,
            virtual_maps=None,
            data_keys=data_keys,
            data_registerIDs={k: data_registerIDs[k] for k in data_keys},
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


################################################################################

'''
MultiDevice and preset subclasses.
--------------------------------

Description:
    MultiDevice extends MultiSource for HARP multi-device data.
    Creates a HarpDevice internally to load dfs_dict, then passes
    it to MultiSource for virtual array construction.

    Subclasses provide preset configurations for specific multi-device
    setups (Nosepoke, BehaviorNosepoke).

Contents:
--------------------------------
- MultiDevice(MultiSource)
    HARP-specific multi-device orchestrator.
- Nosepoke(MultiDevice)
    Preset with default data_keys for nosepoke peripherals.
- BehaviorNosepoke(MultiDevice)
    Preset that auto-builds virtual_maps from channel_list and
    channel_type config (backwards-compatible interface).
'''


################################################################################
# Imports
################################################################################

from pathlib import Path

from data_conduit.datasource import DataSource
from data_conduit.datasource.devices import HarpDevice
from data_conduit.multisource import MultiSource

################################################################################




################################################################################
# MultiDevice
################################################################################

class MultiDevice(MultiSource):
    '''
    MultiSource for HARP multi-device binary data.

    Creates a HarpDevice internally to load dfs_dict from the 
    experiment directory, then delegates to MultiSource for the 
    virtual array construction pipeline.

    Parameters
    ----------
    virtual_maps : dict
        Mapping of data_keys to virtual maps.
        e.g. {'Activations': {'NP0': {'device': 'Behavior0', 
              'register': '32', 'localID': 'DIPort0'}, ...}}
    experiment_directory_path : str or Path
        Path to the experiment directory.
    harp_device_yaml_path : str or Path
        Path to the HARP device YAML configuration file.
    device_type : str
        Device prefix for folder filtering. Default 'Behavior'.
    device_list : list or None
        List of expected device names.
    device_IDs : dict or None
        Mapping of device names to IDs.
    device_registers_dict : dict or None
        Per-device register filtering.
        e.g. {'Behavior0': ['8', '32', '34'], ...}
    data_keys : list[str] or None
        Data keys to process. Defaults to virtual_maps.keys().
    global_coord_name : str
        Name of the global coordinate dimension.
    virtual_coord_names : list[str] or None
        Names of virtual coordinate dimensions.
    dict_of : str
        Format of virtual_map entries: 'dicts' or 'tuples'.
    data_array_names : dict or None
        Custom DataArray names per data_key.
    data_array_attrs : dict or None
        Attributes for DataArrays.
    lookup_array_names : dict or None
        Custom lookup array names per data_key.
    lookup_array_attrs : dict or None
        Attributes for lookup arrays.
    test_values : bool
        If True, use human-readable test values.
    fill_value : any
        Fill value for missing entries.
    verbose : bool
        If True, print progress.

    Attributes
    ----------
    harp_device : HarpDevice
        The underlying HarpDevice instance.
    dfs_dict : dict
        Nested dict of DataFrames from HarpDevice.
    data_arrays : dict[str, xr.DataArray]
        Constructed DataArrays per data_key.
    lookup_arrays : dict[str, xr.DataArray]
        Constructed lookup arrays per data_key.
    lookup_virtual_coords : dict[str, dict]
        Unique virtual coordinate values per data_key.
    '''

    def __init__(self,
                 # Virtual map parameters
                 virtual_maps: dict,
                 # HarpDevice parameters
                 experiment_directory_path: str | Path,
                 harp_device_yaml_path: str | Path = './device.yml',
                 device_type: str = 'Behavior',
                 device_list: list | None = None,
                 device_IDs: dict | None = None,
                 device_registers_dict: dict | None = None,
                 # MultiSource parameters
                 data_keys: list | None = None,
                 global_coord_name: str = 'global_coord',
                 virtual_coord_names: list | None = None,
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
        '''
        Initialise MultiDevice.

        Creates a HarpDevice to load dfs_dict, then delegates to
        MultiSource for virtual array construction.
        '''
        self.harp_device_yaml_path = harp_device_yaml_path
        self.device_type = device_type
        self.device_list = device_list
        self.device_IDs = device_IDs
        self.device_registers_dict = device_registers_dict

        #=== i| Create HarpDevice to load data
        if verbose:
            print("Creating HarpDevice to load data...")

        self.harp_device = HarpDevice(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            verbose=verbose,
            **kwargs,
        )

        #=== ii| Delegate to MultiSource with the loaded dfs_dict
        super().__init__(
            dfs_dict=self.harp_device.dfs_dict,
            virtual_maps=virtual_maps,
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




################################################################################
# Nosepoke
################################################################################

class Nosepoke(MultiDevice):
    '''
    Preset MultiDevice for nosepoke peripheral data.

    Sets default global_coord_name, virtual_coord_names, and data_keys
    for the standard nosepoke setup.

    Parameters
    ----------
    virtual_maps : dict
        Mapping of data_keys to virtual maps.
    experiment_directory_path : str or Path
        Path to the experiment directory.
    harp_device_yaml_path : str or Path
        Path to the HARP device YAML.
    device_list : list or None
        List of device names.
    device_registers_dict : dict or None
        Per-device register filtering.
    data_keys : list
        Data categories. Default ['Activations', 'LEDs', 'Valves', 'Rewards'].
    global_coord_name : str
        Default 'peripherals'.
    virtual_coord_names : list
        Default ['device', 'register', 'localID'].
    verbose : bool
        If True, print progress.
    **kwargs
        Additional kwargs passed to MultiDevice/HarpDevice.
    '''

    def __init__(self,
                 virtual_maps: dict,
                 experiment_directory_path: str | Path,
                 harp_device_yaml_path: str | Path = './device.yml',
                 device_type: str = 'Behavior',
                 device_list: list | None = None,
                 device_IDs: dict | None = None,
                 device_registers_dict: dict | None = None,
                 data_keys: list | None = None,
                 global_coord_name: str = 'peripherals',
                 virtual_coord_names: list | None = None,
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
        '''Initialise Nosepoke with nosepoke-specific defaults.'''

        if data_keys is None:
            data_keys = ['Activations', 'LEDs', 'Valves', 'Rewards']
        if virtual_coord_names is None:
            virtual_coord_names = ['device', 'register', 'localID']

        super().__init__(
            virtual_maps=virtual_maps,
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
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
            **kwargs,
        )


################################################################################




################################################################################
# BehaviorNosepoke
################################################################################

class BehaviorNosepoke(MultiDevice):
    '''
    Preset MultiDevice that auto-builds virtual_maps from channel config.

    Translates the old nosepoke-specific terminology into the generic
    MultiDevice framework by building virtual_maps from channel_list,
    type_keys, channel_type_localIDs, and channel_type_registerIDs.

    Channel assignment uses device-major order:
        channel_list[0] → device_list[0], localIDs[0]
        channel_list[1] → device_list[0], localIDs[1]
        channel_list[2] → device_list[0], localIDs[2]
        channel_list[3] → device_list[1], localIDs[0]
        ...

    Parameters
    ----------
    channel_list : list[str]
        Ordered global channel names. e.g. ['NP_0', 'NP_1', ..., 'NP_17']
    type_keys : list[str]
        Data categories. Maps to data_keys.
        Default ['Activations', 'LEDs', 'Valves', 'Rewards'].
    channel_type_localIDs : dict[str, list[str]]
        Per-type_key ordered list of localIDs.
        e.g. {'Activations': ['DIPort0', 'DIPort1', 'DIPort2']}
    channel_type_registerIDs : dict[str, dict[str, str]]
        Per-type_key mapping of localID → register address.
        e.g. {'Activations': {'DIPort0': '32', 'DIPort1': '32', ...}}
    experiment_directory_path : str or Path
        Path to the experiment directory.
    harp_device_yaml_path : str or Path
        Path to the HARP device YAML.
    device_list : list
        List of device names.
    device_registers_dict : dict
        Per-device register filtering.
    virtual_maps : dict or None
        If provided directly, skips auto-building from channel config.
    verbose : bool
        If True, print progress.
    **kwargs
        Additional kwargs passed to MultiDevice/HarpDevice.

    Usage
    -----
    >>> bnp = BehaviorNosepoke(
    ...     channel_list=['NP_0', ..., 'NP_17'],
    ...     type_keys=['Activations', 'LED'],
    ...     channel_type_localIDs={
    ...         'Activations': ['DIPort0', 'DIPort1', 'DIPort2'],
    ...         'LED': ['DOPort0', 'DOPort1', 'DOPort2'],
    ...     },
    ...     channel_type_registerIDs={
    ...         'Activations': {'DIPort0': '32', ...},
    ...         'LED': {'DOPort0': '34', ...},
    ...     },
    ...     experiment_directory_path='/path/to/experiment',
    ...     harp_device_yaml_path='/path/to/device.yml',
    ...     device_list=['Behavior0', 'Behavior1', ...],
    ...     device_registers_dict={...},
    ... )
    >>> bnp.data_arrays['Activations']
    '''

    def __init__(self,
                 # Nosepoke-specific parameters (used to build virtual_maps)
                 channel_list: list[str],
                 type_keys: list | None = None,
                 channel_type_localIDs: dict[str, list[str]] | None = None,
                 channel_type_registerIDs: dict[str, dict[str, str]] | None = None,
                 # HarpDevice parameters
                 experiment_directory_path: str | Path = None,
                 harp_device_yaml_path: str | Path = './device.yml',
                 device_type: str = 'Behavior',
                 device_list: list | None = None,
                 device_IDs: dict | None = None,
                 device_registers_dict: dict | None = None,
                 # MultiSource parameters
                 virtual_maps: dict | None = None,
                 global_coord_name: str = 'peripherals',
                 virtual_coord_names: list | None = None,
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
        '''
        Initialise BehaviorNosepoke.

        Auto-builds virtual_maps from channel config if not provided
        directly, then delegates to MultiDevice.
        '''
        if type_keys is None:
            type_keys = ['Activations', 'LEDs', 'Valves', 'Rewards']
        if virtual_coord_names is None:
            virtual_coord_names = ['device', 'register', 'localID']

        #=== Store nosepoke-specific references
        self.channel_list = channel_list
        self.type_keys = type_keys
        self.channel_type_localIDs = channel_type_localIDs
        self.channel_type_registerIDs = channel_type_registerIDs

        #=== Build virtual_maps from nosepoke config if not provided
        if virtual_maps is None:
            if not all(v is not None for v in [
                channel_list, type_keys, channel_type_localIDs,
                channel_type_registerIDs, device_list
            ]):
                raise ValueError(
                    "When virtual_maps is not provided, all of "
                    "channel_list, type_keys, channel_type_localIDs, "
                    "channel_type_registerIDs, and device_list must "
                    "be provided to auto-build virtual_maps."
                )

            virtual_maps = self._build_virtual_maps(
                channel_list=channel_list,
                type_keys=type_keys,
                channel_type_localIDs=channel_type_localIDs,
                channel_type_registerIDs=channel_type_registerIDs,
                device_list=device_list,
                verbose=verbose,
            )

        #=== Delegate to MultiDevice
        super().__init__(
            virtual_maps=virtual_maps,
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            data_keys=type_keys,
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


    @staticmethod
    def _build_virtual_maps(channel_list: list[str],
                            type_keys: list[str],
                            channel_type_localIDs: dict[str, list[str]],
                            channel_type_registerIDs: dict[str, dict[str, str]],
                            device_list: list[str],
                            verbose: bool = False,
                            ) -> dict:
        '''
        Build virtual_maps from nosepoke channel configuration.

        Assigns channels in device-major order:
            channel_list[i] → device_list[i // n_locals], 
                              localIDs[i % n_locals]

        Parameters
        ----------
        channel_list : list[str]
            Ordered global channel names.
        type_keys : list[str]
            Data categories to process.
        channel_type_localIDs : dict
            Per-type_key ordered list of localIDs.
        channel_type_registerIDs : dict
            Per-type_key mapping of localID → register address.
        device_list : list[str]
            List of device names.
        verbose : bool
            If True, print progress.

        Returns
        -------
        dict
            {type_key: {channel: {'device': ..., 'register': ..., 
            'localID': ...}, ...}, ...}
        '''
        virtual_maps = {}

        for tk in type_keys:
            local_ids = channel_type_localIDs[tk]
            reg_map = channel_type_registerIDs[tk]
            n_locals = len(local_ids)

            if len(channel_list) % n_locals != 0:
                raise ValueError(
                    f"len(channel_list)={len(channel_list)} is not a "
                    f"multiple of len(channel_type_localIDs['{tk}'])"
                    f"={n_locals}."
                )
            n_devices_needed = len(channel_list) // n_locals
            if n_devices_needed > len(device_list):
                raise ValueError(
                    f"channel_list requires {n_devices_needed} devices "
                    f"for type_key '{tk}', but only {len(device_list)} "
                    f"provided."
                )

            vmap = {}
            for i, channel in enumerate(channel_list):
                dev_idx = i // n_locals
                lid = local_ids[i % n_locals]
                vmap[channel] = {
                    'device': device_list[dev_idx],
                    'register': reg_map[lid],
                    'localID': lid,
                }
            virtual_maps[tk] = vmap

        if verbose:
            print(
                f"BehaviorNosepoke: Built virtual_maps for "
                f"{list(virtual_maps.keys())} from channel_list "
                f"({len(channel_list)} channels) × device_list "
                f"({len(device_list)} devices)."
            )

        return virtual_maps


################################################################################
    """
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

from data_conduit.harptools import collect_harp_dfs
from data_conduit.multisource.multisource_core import MultiSource

################################################################################



################################################################################
# MultiDevice
################################################################################


class MultiDevice(MultiSource):
    '''
    Device-oriented MultiSource wrapper.

    Like Device, this class focuses on obtaining the right dfs_dict shape.
    Unlike Device, it then delegates to MultiSource to construct virtual-map
    aligned outputs (data_arrays, lookup_arrays, lookup_virtual_coords).

    Parameters 
    ----------
    experiment_directory_path : str or Path
        Root directory to walk for HARP data.
    harp_device_yaml_path : str or Path
        Path to HARP device YAML for parsing device/register metadata.
    device_type : str
        Device type to filter for in the HARP device YAML.
    device_list : list of str
        List of device names to include. If None, include all devices of the specified type.
    device_IDs : dict
        Optional mapping of device names to device IDs for filtering.
    device_registers : dict
        Optional mapping of device names to lists of register IDs to include.
    virtual_maps : dict or None
        Optional mapping of virtual map names to virtual maps for lookup construction.
    data_keys : list or None
        Optional list of keys in virtual_maps to use for data array construction. If None, use all keys.
    global_coord_name : str
        Name of the global coordinate dimension in the lookup array.
    virtual_coord_names : list of str
        Names of the virtual coordinate dimensions in the lookup array.
    dict_of : str
        Format for virtual_map input: 'dicts' for dict-of-dicts, 'tuples' for dict-of-tuples.
    data_array_names : dict or None
        Optional custom names for data arrays keyed by data_key.
        e.g. {'Activations': 'my_activations_data'}
    data_array_attrs : dict or None
        Optional dict of attributes to set on constructed data arrays.
    rekey : bool
        If True, rekey the dfs_dict to use device names instead of IDs for easier navigation. If False, keep original keys. Default is True.
    lookup_array_names : dict or None
        Optional custom names for lookup arrays keyed by data_key.
        e.g. {'Activations': 'my_activations_lookup'}
    lookup_array_attrs : dict or None
        Optional dict of attributes to set on constructed lookup arrays.
    test_values : bool
        If True, populate arrays with human-readable debug strings instead of NaNs.
    fill_value : any
        Value to use for filling missing entries in the lookup array. Default is None.
    verbose : bool
        If True, print detailed information during loading and construction.
    **kwargs
        Additional keyword arguments passed to collect_harp_dfs and MultiSource.
        e.g. l0_selector, l1_selector for collect_harp_dfs; any MultiSource kwargs for output construction.
    
    Attributes
    ----------
    dfs_dict : dict
        Nested dictionary of DataFrames from collect_harp_dfs.
    data_arrays : dict[str, xr.DataArray]
        Named DataArrays built from virtual maps.
    lookup_arrays : dict[str, xr.DataArray]
        Named lookup DataArrays built from virtual maps.
    lookup_virtual_coords : dict[str, list]
        Virtual coordinate names for each lookup array.


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
        rekey: bool = True,
        #== multisource config ==#
        global_coord_name: str = 'global_coord',
        virtual_coord_names: list[str] | None = ['device', 'register', 'localID'],
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

        if dfs_dict is None and experiment_directory_path is not None:
            dfs_dict = collect_harp_dfs(
                base_path=experiment_directory_path,
                harp_device_yaml_path=harp_device_yaml_path,
                device_type=device_type,
                device_registers=device_registers,
                device_list=device_list,
                rekey=rekey,
                verbose=verbose,
            )

        super().__init__(
            dfs_dict=dfs_dict,
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
# Nosepoke Preset
################################################################################
class Nosepoke(MultiDevice):
    '''
    Preset MultiDevice for common nosepoke peripheral data configuration.

    Pre-configures a 6-board, 18-nosepoke layout with 4 data types
    (Activations, LEDs, Valves, Rewards). Each board has 3 nosepokes.
    All defaults can be overridden.

    Parameters
    ----------
    experiment_directory_path : str or Path or None
        Root path for loading from disk.
    harp_device_yaml_path : str or Path
        Path to HARP device YAML schema.
    device_list : list[str]
        Device names. Default: ['Behavior0', ..., 'Behavior5'].
    device_registers : dict
        Registers per device. Default: {'Behavior0': ['32', '34'], ...}.
    virtual_maps : dict
        Virtual maps for all data keys. Default maps NP_0..NP_17 across
        6 Behavior boards with 3 ports each per data key.
    data_keys : list[str]
        Data keys to construct. Default: ['Activations', 'LEDs', 'Valves', 'Rewards'].
    global_coord_name : str
        Name of the global coordinate dimension. Default: 'peripherals'.
    virtual_coord_names : list[str]
        Virtual coordinate names. Default: ['device', 'register', 'localID'].
    rekey : bool
        If True, rekey HARP file stems to register addresses. Default: True.
    verbose : bool
        If True, print progress during loading and construction.
    **kwargs
        Passed through to MultiDevice/MultiSource (data_array_names,
        data_array_attrs, lookup_array_names, lookup_array_attrs,
        test_values, fill_value, dict_of).

    Attributes
    ----------
    dfs_dict : dict
        Nested dictionary of DataFrames from collect_harp_dfs.
    data_arrays : dict[str, xr.DataArray]
        One DataArray per data key (Activations, LEDs, Valves, Rewards),
        each with dims [Time × peripherals] and virtual coords attached.
        Queryable via da.ulookup.select(device='Behavior0').
    lookup_arrays : dict[str, xr.DataArray]
        One lookup array per data key.
    lookup_virtual_coords : dict[str, dict]
        Unique virtual coordinate values for each data key.
    '''

    def __init__(
        self,
        experiment_directory_path: str | Path | None = None,
        harp_device_yaml_path: str | Path = './device.yml',
        device_list: list[str] = [f'Behavior{i}' for i in range(6)],
        device_registers: dict = {f'Behavior{i}': ['32', '34'] for i in range(6)},
        virtual_maps: dict = {
            'Activations': {f'NP_{i}': {'device': f'Behavior{i//3}', 'register': '32', 'localID': ['DIPort0','DIPort1','DIPort2'][i%3]} for i in range(18)},
            'LEDs':        {f'NP_{i}': {'device': f'Behavior{i//3}', 'register': '34', 'localID': ['DOPort0','DOPort1','DOPort2'][i%3]} for i in range(18)},
            'Valves':      {f'NP_{i}': {'device': f'Behavior{i//3}', 'register': '34', 'localID': ['DOPort3','DOPort4','DOPort5'][i%3]} for i in range(18)},
            'Rewards':     {f'NP_{i}': {'device': f'Behavior{i//3}', 'register': '32', 'localID': ['DIPort3','DIPort4','DIPort5'][i%3]} for i in range(18)},
        },
        data_keys: list[str] = ['Activations', 'LEDs', 'Valves', 'Rewards'],
        global_coord_name: str = 'peripherals',
        virtual_coord_names: list[str] = ['device', 'register', 'localID'],
        rekey: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        '''Initialise Nosepoke preset and delegate to MultiDevice.'''
        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type='Behavior',
            device_list=device_list,
            device_registers=device_registers,
            rekey=rekey,
            virtual_maps=virtual_maps,
            data_keys=data_keys,
            global_coord_name=global_coord_name,
            virtual_coord_names=virtual_coord_names,
            verbose=verbose,
            **kwargs,
        )

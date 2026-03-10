'''
HarpDevice and preset subclasses.
--------------------------------

Description:
    HarpDevice extends DataSource for HARP binary data. Handles 
    double-folder collapsing, rekeying file stems to register 
    addresses, and per-device register filtering.

Contents:
--------------------------------
- HarpDevice(DataSource)
- SoundCard(HarpDevice)
- CameraStart(HarpDevice)
- Camera0Frames(HarpDevice)
- AnalogSync(HarpDevice)
'''


################################################################################
# Imports
################################################################################

from pathlib import Path

from data_conduit.datasource import DataSource
from data_conduit.harptools import read_harp_bin
from data_conduit.utils import starts_with

################################################################################




################################################################################
# HarpDevice
################################################################################

class HarpDevice(DataSource):
    '''
    DataSource for HARP binary device data.

    Extends DataSource to handle HARP-specific concerns:
    - Passes read_harp_bin as the .bin reader
    - Collapses double folders (Behavior0/Behavior0/ → Behavior0/)
    - Rekeys file stems to register addresses
    - Filters per device_registers_dict

    Parameters
    ----------
    experiment_directory_path : str or Path
        Path to the experiment directory.
    harp_device_yaml_path : str or Path
        Path to the YAML configuration file for the device.
    device_type : str
        Device prefix to filter folders (e.g. 'Behavior', 'SoundCard').
    device_list : list or None
        List of expected device names.
    device_IDs : dict or None
        Dictionary mapping device names to their expected IDs.
    device_registers_dict : dict or None
        Per-device register filtering.
        e.g. {'Behavior0': ['8', '34'], 'Behavior1': ['8', '44']}
    da_type_dict : dict or None
        Mapping of friendly names to (device, register) paths.
        e.g. {'PlaySoundFreq': ('SoundCard', '32')}
    verbose : bool
        If True, print warnings during processing.
    **kwargs
        Additional level selectors passed to collect_dfs.
    '''

    def __init__(self,
                 experiment_directory_path=None,
                 harp_device_yaml_path='./device.yml',
                 device_type='Behavior',
                 device_list=None,
                 device_IDs=None,
                 device_registers_dict=None,
                 da_type_dict=None,
                 verbose=False,
                 **kwargs,
                 ):
        '''
        Initialise DataSource without preset data. Optionally filter to only include timestamps present in ALL devices.
        
        Parameters
        ----------
        experiment_directory_path : str or Path
            Path to the experiment directory.
        harp_device_yaml_path : str or Path
            Path to the YAML configuration file for the device.
        device_type : str
            Device prefix to filter folders (e.g. 'Behavior', 'SoundCard').
        device_list : list or None
            List of expected device names.
        device_IDs : dict or None
            Dictionary mapping device names to their expected IDs.
        device_registers_dict : dict or None
            Per-device register filtering.
            e.g. {'Behavior0': ['8', '34'], 'Behavior1': ['8', '44']}
        da_type_dict : dict or None
            Mapping of friendly names to (device, register) paths.
            e.g. {'PlaySoundFreq': ('SoundCard', '32')}
        matching_only : bool
            If True, filter each DataArray to only include timestamps present in ALL devices.
        verbose : bool
            If True, print warnings during processing.
        **kwargs
            Additional level selectors passed to collect_dfs.
        '''
        self.harp_device_yaml_path = harp_device_yaml_path
        self.device_type = device_type
        self.device_list = device_list
        self.device_IDs = device_IDs
        self.device_registers_dict = device_registers_dict

        #=== i| Load via DataSource (no name_map yet — need to post-process first)
        super().__init__(
            experiment_directory_path=experiment_directory_path,
            readers={'.bin': read_harp_bin},
            reader_kwargs={'.bin': {'harp_device_yaml_path': harp_device_yaml_path}},
            verbose=verbose,
            l0_selector=starts_with(device_type),
            **kwargs,
        )

        #=== ii| Collapse double folders
        #    {Behavior0: {Behavior0: {stem: df}}} → {Behavior0: {stem: df}}
        for key in list(self.dfs_dict.keys()):
            value = self.dfs_dict[key]
            if isinstance(value, dict) and list(value.keys()) == [key]:
                self.dfs_dict[key] = value[key]

        #=== iii| Rekey file stems → register addresses
        #    'Behavior0_8_1904-01-01T02-00-00' → '8'
        for device in list(self.dfs_dict.keys()):
            if isinstance(self.dfs_dict[device], dict):
                rekeyed = {}
                for stem, df in self.dfs_dict[device].items():
                    try:
                        addr = stem.split('_')[1]
                        rekeyed[addr] = df
                    except (IndexError, ValueError):
                        rekeyed[stem] = df
                self.dfs_dict[device] = rekeyed

        #=== iv| Filter per device_registers_dict
        if device_registers_dict is not None:
            filtered = {}
            for device_name, registers in device_registers_dict.items():
                if device_name in self.dfs_dict:
                    allowed = {str(a) for a in registers}
                    filtered[device_name] = {
                        addr: df for addr, df in self.dfs_dict[device_name].items()
                        if addr in allowed
                    }
                elif verbose:
                    print(f"Warning: {device_name} not found in loaded data.")
            self.dfs_dict = filtered

        #=== v| Build named DataArrays now that dfs_dict is in the right shape
        if da_type_dict is not None:
            self._build_data_arrays(da_type_dict)


################################################################################




################################################################################
# SoundCard
################################################################################

class SoundCard(HarpDevice):
    '''Preset config for SoundCard device data.'''

    def __init__(self,
                 experiment_directory_path=None,
                 harp_device_yaml_path='./soundcard.yml',
                 device_type='SoundCard',
                 device_list=None,
                 device_IDs=None,
                 device_registers_dict=None,
                 da_type_dict=None,
                 verbose=False,
                 ):
        '''
        Initialise SoundCard DataSource with preset config for SoundCard data. Optionally filter to only include timestamps present in ALL devices.
        
        Parameters
        ----------
        experiment_directory_path : str or Path
            Path to the experiment directory.
        harp_device_yaml_path : str or Path
            Path to the YAML configuration file for the device.
        device_type : str
            Device prefix to filter folders (e.g. 'Behavior', 'SoundCard').
        device_list : list or None
            List of expected device names.
        device_IDs : dict or None
            Dictionary mapping device names to their expected IDs.
        device_registers_dict : dict or None
            Per-device register filtering.
            e.g. {'Behavior0': ['8', '34'], 'Behavior1': ['8', '44']}
        da_type_dict : dict or None
            Mapping of friendly names to (device, register) paths.
            e.g. {'PlaySoundFreq': ('SoundCard', '32')}
        matching_only : bool
            If True, filter each DataArray to only include timestamps present in ALL devices.
        verbose : bool
            If True, print warnings during processing.
        **kwargs
            Additional level selectors passed to collect_dfs.
        '''
        if device_list is None:
            device_list = ['SoundCard']
        if device_IDs is None:
            device_IDs = {'SoundCard': 'SoundCard0'}
        if device_registers_dict is None:
            device_registers_dict = {'SoundCard': ['8', '32', '33', '35']}
        if da_type_dict is None:
            da_type_dict = {
                'PlaySoundFreq':    ('SoundCard', '32'),
                'StopLog':          ('SoundCard', '33'),
                'AttenuationRight': ('SoundCard', '35'),
            }

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            da_type_dict=da_type_dict,
            verbose=verbose,
        )


################################################################################




################################################################################
# CameraStart
################################################################################

class CameraStart(HarpDevice):
    '''Preset config for CameraStart device data.'''

    def __init__(self,
                 experiment_directory_path=None,
                 harp_device_yaml_path='./device.yml',
                 device_type='Behavior',
                 device_list=None,
                 device_IDs=None,
                 device_registers_dict=None,
                 da_type_dict=None,
                 verbose=False,
                 ):
        '''
        Initialise CameraStart DataSource with preset config for CameraStart data. Optionally filter to only include timestamps present in ALL devices.
        
        Parameters
        ----------
        experiment_directory_path : str or Path
            Path to the experiment directory.
        harp_device_yaml_path : str or Path
            Path to the YAML configuration file for the device.
        device_type : str
            Device prefix to filter folders (e.g. 'Behavior', 'SoundCard').
        device_list : list or None
            List of expected device names.
        device_IDs : dict or None
            Dictionary mapping device names to their expected IDs.
        device_registers_dict : dict or None
            Per-device register filtering.
            e.g. {'Behavior0': ['8', '34'], 'Behavior1': ['8', '44']}
        da_type_dict : dict or None
            Mapping of friendly names to (device, register) paths.
            e.g. {'PlaySoundFreq': ('SoundCard', '32')}
        matching_only : bool
            If True, filter each DataArray to only include timestamps present in ALL devices.
        verbose : bool
            If True, print warnings during processing.
        **kwargs
            Additional level selectors passed to collect_dfs.
        '''
        if device_list is None:
            device_list = [f'Behavior{i}' for i in range(6)]
        if device_IDs is None:
            device_IDs = {f'Behavior{i}': f'ID_{i}' for i in range(6)}
        if device_registers_dict is None:
            device_registers_dict = {f'Behavior{i}': ['78'] for i in range(6)}
        if da_type_dict is None:
            da_type_dict = {
                f'Behavior{i}_StartCameras': (f'Behavior{i}', '78')
                for i in range(6)
            }

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            da_type_dict=da_type_dict,
            verbose=verbose,
        )


################################################################################




################################################################################
# Camera0Frames
################################################################################

class Camera0Frames(HarpDevice):
    '''Preset config for Camera0Frame device data.'''

    def __init__(self,
                 experiment_directory_path=None,
                 harp_device_yaml_path='./device.yml',
                 device_type='Behavior',
                 device_list=None,
                 device_IDs=None,
                 device_registers_dict=None,
                 da_type_dict=None,
                 matching_only=False,
                 verbose=False,
                 ):
        '''
        Initialise Camera0Frames DataSource with preset config for Camera0Frame data. Optionally filter to only include timestamps present in ALL devices.
        
        Parameters
        ----------
        experiment_directory_path : str or Path
            Path to the experiment directory.
        harp_device_yaml_path : str or Path
            Path to the YAML configuration file for the device.
        device_type : str
            Device prefix to filter folders (e.g. 'Behavior', 'SoundCard').
        device_list : list or None
            List of expected device names.
        device_IDs : dict or None
            Dictionary mapping device names to their expected IDs.
        device_registers_dict : dict or None
            Per-device register filtering.
            e.g. {'Behavior0': ['8', '34'], 'Behavior1': ['8', '44']}
        da_type_dict : dict or None
            Mapping of friendly names to (device, register) paths.
            e.g. {'PlaySoundFreq': ('SoundCard', '32')}
        matching_only : bool
            If True, filter each DataArray to only include timestamps present in ALL devices.
        verbose : bool
            If True, print warnings during processing.
        **kwargs
            Additional level selectors passed to collect_dfs.
        '''
        if device_list is None:
            device_list = [f'Behavior{i}' for i in range(6)]
        if device_IDs is None:
            device_IDs = {f'Behavior{i}': f'ID_{i}' for i in range(6)}
        if device_registers_dict is None:
            device_registers_dict = {f'Behavior{i}': ['92'] for i in range(6)}
        if da_type_dict is None:
            da_type_dict = {
                f'Behavior{i}_Camera0Frame': (f'Behavior{i}', '92')
                for i in range(6)
            }

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            da_type_dict=da_type_dict,
            verbose=verbose,
        )

        if matching_only:
            self._filter_to_matching_times()

    def _filter_to_matching_times(self):
        '''Filter each DataArray to only include timestamps present in ALL devices.'''
        if not self.data_arrays:
            return

        time_sets = [set(da.get_index('Time')) for da in self.data_arrays.values()]
        common = sorted(set.intersection(*time_sets)) if time_sets else []

        for key in self.data_arrays:
            self.data_arrays[key] = self.data_arrays[key].sel(Time=common)

        if self.verbose:
            print(f"Filtered to {len(common)} matching timestamps.")


################################################################################




################################################################################
# AnalogSync
################################################################################

class AnalogSync(HarpDevice):
    '''Preset config for AnalogSync device data.'''

    def __init__(self,
                 experiment_directory_path=None,
                 harp_device_yaml_path='./device.yml',
                 device_type='Behavior',
                 device_list=None,
                 device_IDs=None,
                 device_registers_dict=None,
                 da_type_dict=None,
                 matching_only=False,
                 verbose=False,
                 ):
        '''
        Initialise Camera0Frames DataSource with preset config for Camera0Frame data. Optionally filter to only include timestamps present in ALL devices.
        
        Parameters
        ----------
        experiment_directory_path : str or Path
            Path to the experiment directory.
        harp_device_yaml_path : str or Path
            Path to the YAML configuration file for the device.
        device_type : str
            Device prefix to filter folders (e.g. 'Behavior', 'SoundCard').
        device_list : list or None
            List of expected device names.
        device_IDs : dict or None
            Dictionary mapping device names to their expected IDs.
        device_registers_dict : dict or None
            Per-device register filtering.
            e.g. {'Behavior0': ['8', '34'], 'Behavior1': ['8', '44']}
        da_type_dict : dict or None
            Mapping of friendly names to (device, register) paths.
            e.g. {'PlaySoundFreq': ('SoundCard', '32')}
        matching_only : bool
            If True, filter each DataArray to only include timestamps present in ALL devices.
        verbose : bool
            If True, print warnings during processing.
        **kwargs
            Additional level selectors passed to collect_dfs.
        '''

        if device_list is None:
            device_list = [f'Behavior{i}' for i in range(6)]
        if device_IDs is None:
            device_IDs = {f'Behavior{i}': f'ID_{i}' for i in range(6)}
        if device_registers_dict is None:
            device_registers_dict = {f'Behavior{i}': ['44'] for i in range(6)}
        if da_type_dict is None:
            da_type_dict = {
                f'Behavior{i}_AnalogData': (f'Behavior{i}', '44')
                for i in range(6)
            }

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            da_type_dict=da_type_dict,
            verbose=verbose,
        )

        if matching_only:
            self._filter_to_matching_times()

    def _filter_to_matching_times(self):
        '''Filter each DataArray to only include timestamps present in ALL devices.'''
        if not self.data_arrays:
            return

        time_sets = [set(da.get_index('Time')) for da in self.data_arrays.values()]
        common = sorted(set.intersection(*time_sets)) if time_sets else []

        for key in self.data_arrays:
            self.data_arrays[key] = self.data_arrays[key].sel(Time=common)

        if self.verbose:
            print(f"Filtered to {len(common)} matching timestamps.")


################################################################################
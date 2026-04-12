'''
Utility functions for devices.


'''
########################################################################################################################
# Imports
########################################################################################################################

import os
import pandas as pd
import numpy as np
import xarray as xr
import json
import yaml
from collections.abc import Mapping
from pathlib import Path

from Refactor.HarpExtender.harp_extender_core import collect_device_folders
########################################################################################################################








########################################################################################################################
# Private Helper Functions
########################################################################################################################




#===============================================================================
# 1| Collect Filetype Path
#===============================================================================



#===============================================================================




#===============================================================================
# 2|
#===============================================================================



#===============================================================================






########################################################################################################################








########################################################################################################################
# Public Functions
########################################################################################################################




################################################################################
# Filetype Data Base Class
################################################################################
class Device:
    '''
    Base class for devices. Use for getting data arrays from device registers.
    '''
    def __init__(self,
                 experiment_directory_path: str | Path = None,                  # Path to experimental directory (str or Path)
                 harp_device_yaml_path: str | Path = None,                      # Path to harp_device yaml file (str or Path). If None, will attempt to find in experimental_directory_path.
                 device_type: str = None,
                 device_list: list | None = None,
                 device_IDs: dict | None = None,
                 device_registers_dict: dict | None = None,
                 da_type_dict: dict | None = None,
                 lookfor: str | None = None,
                 validate: bool = False,
                 verbose: bool = False
    ):
        
        super(Device, self).__init__()

        
        ##===== User-Defined Parameters ====================================
        self.experiment_directory_path  = experiment_directory_path
        self.harp_device_yaml_path     = harp_device_yaml_path
        self.device_type               = device_type
        self.device_list               = device_list
        self.device_IDs                = device_IDs
        self.device_registers_dict     = device_registers_dict
        self.da_type_dict              = da_type_dict
        self.lookfor                   = lookfor
        self.validate                  = validate
        self.verbose                   = verbose

        ##===== Setup ======================================================
        self.harp_device_reader = self._construct_device_reader()
        self.device_folders     = self._collect_device_folders(verbose=self.verbose)
        self.device_dfs_dict    = self._collect_device_dfs(verbose=self.verbose)

        ##===== Build DataArrays from da_type_dict =========================
        self.device_da_dict = {}
        if self.da_type_dict is not None:
            for da_name, (device_name, register_addr) in self.da_type_dict.items():
                self.device_da_dict[da_name] = (
                    self.device_dfs_dict[device_name][register_addr].to_xarray()
                )
                if self.verbose:
                    print(f"Built DataArray '{da_name}' from {device_name}[{register_addr}]")

    #===========================================================================
    # 1| Construct Device Reader
    #===========================================================================
    def _construct_device_reader(self, 
                                harp_device_yaml_path: str | None = None,
                                ):
        '''
        Wrapper for construct_device_reader function from harp_extender_core. 
        By default, uses self.harp_device_yaml_path. Can be used to construct a device reader with a different yaml path if desired.
        -----------------
        
        Parameters:
            harp_device_yaml_path (str or Path): Path to the YAML configuration file for the device.

        Returns:
            harp.Reader: A harp reader object for the specified device.
        '''
        from Refactor.HarpExtender.harp_extender_core import construct_device_reader
        
        if harp_device_yaml_path is not None:
            return construct_device_reader(harp_device_yaml_path)
        else:
            return construct_device_reader(self.harp_device_yaml_path)
    #===========================================================================
    
    
    #===========================================================================
    # 2| Collect Device Folders
    #===========================================================================
    def _collect_device_folders(self,
                               experiment_directory_path: str | None = None,
                               device_type: str | None = None,
                               verbose: bool | None = None,
                               ):
        '''
        Wrapper for collect_device_folders function from harp_extender_core.
        By default, uses self.experiment_directory_path, self.device_type, and self.verbose. 
        -------------------
        
        Parameters:
            experiment_directory_path (str or Path): Path to the experiment directory.
            device_type (str): The type of device to filter folders by (e.g., 'Behavior').

        Returns:
            sorted([f for f in folder_path.glob(f'{device_type}*') if f.is_dir()], key=lambda p: p.name): A list of folder names that match the specified device type.
        '''
        from Refactor.HarpExtender.harp_extender_core import collect_device_folders
        if experiment_directory_path is None:
            experiment_directory_path = self.experiment_directory_path
        if device_type is None:
            device_type = self.device_type
        if verbose is None:
            verbose = self.verbose

        
        return collect_device_folders(experiment_directory_path, device_type)
    #===========================================================================



    #===========================================================================
    # 3| Collect DataFrames
    #===========================================================================
    def _collect_device_dfs(self,
                           experiment_directory_path: str | Path = None,
                           harp_device_yaml_path: str | Path = None,
                           device_type: str = None,
                           device_list: list | None = None,
                           device_IDs: dict | None = None,
                           device_registers_dict: dict | None = None,
                           register_address: int | str = None,
                           register_name: str = None,
                           lookfor: str = None,
                           flatten: bool = False,
                           Return: bool = False,
                           verbose: bool | None = None,
                           validate: bool= False,
                         ) -> dict | None:
        '''
        Wrapper for collect_device_dfs function from harp_extender_core.
        -------------------------
        Modes:
        1. Default (like 5b): Returns nested dict {device: {register: df}}
            Usage: Provide 'device_registers_dict'.
        2. Flat (like 5a): Returns flat dict {device: df}
            Usage: Set 'flatten=True' AND provide 'register_address' OR 'register_name'.
        
        Parameters:
            experiment_directory_path (str or Path): Path to the directory containing experiment logs.
            harp_device_yaml_path (str or Path): Path to the YAML configuration file for the device.
            device_type (str): The type of device to filter folders by (e.g., 'Behavior'). Denotes prefix to match folders.
            device_list (list of str): List of expected device names.
            device_IDs (dict):         device_IDs (dict): Dictionary mapping device names to their expected IDs. {name: ID}.
            device_registers_dict (dict): {device_name: [list_of_registers]}. Dictionary mapping device names to their expected register IDs.
            register_address (int/str): Specific register address to load (overrides dictionary if flattened).
            register_name (str): The name of the register to read. If provided, takes precedence over register_address.
            lookfor (str): If provided, will select register paths containing this substring if multiple possible matches exist.
            flatten (bool): If True, returns a flat dictionary {device: df}. Requires a single target register.
            Return (bool): If True, returns the nested dictionary. If False, only prints progress. (Only applies to validate function)
            verbose (bool): If True, prints additional information during processing.

        Returns:
            dict: Either nested or flat dictionary of DataFrames depending on 'flatten'.
                Nested dict: A nested dictionary where keys are device folder names and values are dictionaries of DataFrames for each register. Structure dict[device_name][register_address][localID]    
                Flattened dict: A dictionary where keys are device folder names and values are DataFrames of the specified register.

        '''
        from Refactor.HarpExtender.harp_extender_core import collect_device_dfs

        # Use self attributes if parameters are not provided
        if device_registers_dict is None and flatten is False:       
                                                device_registers_dict = self.device_registers_dict

        if experiment_directory_path is None:   experiment_directory_path = self.experiment_directory_path
        if harp_device_yaml_path is None:       harp_device_yaml_path = self.harp_device_yaml_path
        if device_type is None:                 device_type = self.device_type
        if device_list is None:                 device_list = self.device_list
        if device_IDs is None:                  device_IDs = self.device_IDs
        if lookfor is None:                     lookfor = self.lookfor
        if verbose is None:                     verbose = self.verbose
        if validate is None:                    validate = self.validate

        
        return collect_device_dfs(experiment_directory_path = experiment_directory_path, 
                                  harp_device_yaml_path = harp_device_yaml_path, 
                                  device_type = device_type, 
                                  device_list = device_list, 
                                  device_IDs = device_IDs, 
                                  device_registers_dict = device_registers_dict, 
                                  register_address = register_address, 
                                  register_name = register_name, 
                                  lookfor = lookfor, 
                                  flatten = flatten, 
                                  Return = Return, 
                                  verbose = verbose, 
                                  validate = validate
                                  )
    
    #===========================================================================


################################################################################




################################################################################
# SoundCard Device Class
################################################################################
class SoundCard(Device):
    '''
    Class to manage loading and processing of SoundCard data from HARP devices.
    Inherits from Device.
    '''

    def __init__(self,
                 experiment_directory_path=None,
                 harp_device_yaml_path='./soundcard.yml',
                 device_type='SoundCard',
                 device_list=None,
                 device_IDs=None,
                 device_registers_dict=None,
                 da_type_dict=None,
                 lookfor=None,
                 validate=False,
                 verbose=False,
                 ):

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

        super(SoundCard, self).__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            da_type_dict=da_type_dict,
            lookfor=lookfor,
            validate=validate,
            verbose=verbose,
        )


################################################################################




################################################################################
# CameraStart Device Class
################################################################################
class CameraStart(Device):
    '''
    Class to manage reading of CameraStart binary files from HARP devices.
    Inherits from Device.
    '''

    def __init__(self,
                 experiment_directory_path=None,
                 harp_device_yaml_path='./device.yml',
                 device_type='Behavior',
                 device_list=None,
                 device_IDs=None,
                 device_registers_dict=None,
                 da_type_dict=None,
                 lookfor=None,
                 validate=False,
                 verbose=False,
                 ):

        if device_list is None:
            device_list = [f'Behavior{i}' for i in range(6)]
        if device_IDs is None:
            device_IDs = {f'Behavior{i}': f'ID_{i}' for i in range(6)}
        if device_registers_dict is None:
            device_registers_dict = {f'Behavior{i}': ['78'] for i in range(6)}
        if da_type_dict is None:
            da_type_dict = {f'Behavior{i}_StartCameras': (f'Behavior{i}', '78') for i in range(6)}

        super(CameraStart, self).__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            da_type_dict=da_type_dict,
            lookfor=lookfor,
            validate=validate,
            verbose=verbose,
        )


################################################################################




################################################################################
# Camera0Frames Device Class
################################################################################
class Camera0Frames(Device):
    '''
    Class to manage reading of Camera0Frame binary files from HARP devices.
    Inherits from Device.
    '''

    def __init__(self,
                 experiment_directory_path=None,
                 harp_device_yaml_path='./device.yml',
                 device_type='Behavior',
                 device_list=None,
                 device_IDs=None,
                 device_registers_dict=None,
                 da_type_dict=None,
                 matching_only=False,
                 lookfor=None,
                 validate=False,
                 verbose=False,
                 ):

        if device_list is None:
            device_list = [f'Behavior{i}' for i in range(6)]
        if device_IDs is None:
            device_IDs = {f'Behavior{i}': f'ID_{i}' for i in range(6)}
        if device_registers_dict is None:
            device_registers_dict = {f'Behavior{i}': ['92'] for i in range(6)}
        if da_type_dict is None:
            da_type_dict = {f'Behavior{i}_Camera0Frame': (f'Behavior{i}', '92') for i in range(6)}

        super(Camera0Frames, self).__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            da_type_dict=da_type_dict,
            lookfor=lookfor,
            validate=validate,
            verbose=verbose,
        )

        if matching_only:
            self._filter_to_matching_entries()

    def _filter_to_matching_entries(self):
        '''Filter each DataArray to only include timestamps present in ALL devices.'''
        if self.verbose:
            print("Warning: _filter_to_matching_entries does not currently work as "
                  "intended due to how timestamps are recorded.")

        device_time_sets = [set(da.get_index('Time')) for da in self.device_da_dict.values()]
        matching_times = set.intersection(*device_time_sets) if device_time_sets else set()

        if self.verbose:
            print(f"Found {len(matching_times)} matching timestamps across all devices.")

        for key, da in self.device_da_dict.items():
            self.device_da_dict[key] = da.sel(Time=sorted(matching_times))


################################################################################




################################################################################
# AnalogSync Device Class
################################################################################
class AnalogSync(Device):
    '''
    Class to manage reading of AnalogSync binary files from HARP devices.
    Inherits from Device.
    '''

    def __init__(self,
                 experiment_directory_path=None,
                 harp_device_yaml_path='./device.yml',
                 device_type='Behavior',
                 device_list=None,
                 device_IDs=None,
                 device_registers_dict=None,
                 da_type_dict=None,
                 matching_only=False,
                 lookfor=None,
                 validate=False,
                 verbose=False,
                 ):

        if device_list is None:
            device_list = [f'Behavior{i}' for i in range(6)]
        if device_IDs is None:
            device_IDs = {f'Behavior{i}': f'ID_{i}' for i in range(6)}
        if device_registers_dict is None:
            device_registers_dict = {f'Behavior{i}': ['44'] for i in range(6)}
        if da_type_dict is None:
            da_type_dict = {f'Behavior{i}_AnalogData': (f'Behavior{i}', '44') for i in range(6)}

        super(AnalogSync, self).__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            da_type_dict=da_type_dict,
            lookfor=lookfor,
            validate=validate,
            verbose=verbose,
        )

        if matching_only:
            self._filter_to_matching_entries()

    def _filter_to_matching_entries(self):
        '''Filter each DataArray to only include timestamps present in ALL devices.'''
        if self.verbose:
            print("Warning: _filter_to_matching_entries does not currently work as "
                  "intended due to how timestamps are recorded.")

        device_time_sets = [set(da.get_index('Time')) for da in self.device_da_dict.values()]
        matching_times = set.intersection(*device_time_sets) if device_time_sets else set()

        if self.verbose:
            print(f"Found {len(matching_times)} matching timestamps across all devices.")

        for key, da in self.device_da_dict.items():
            self.device_da_dict[key] = da.sel(Time=sorted(matching_times))


################################################################################


########################################################################################################################






########################################################################################################################
# Wrappers for renamed, moved, or deprecated functions
########################################################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original

################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original

################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original

################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original

################################################################################





########################################################################################################################

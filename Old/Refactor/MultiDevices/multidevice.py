'''
Uitilities for handling multi-device data arrays.


'''


################################################################################
# Imports
################################################################################

from csv import reader
import os
import re
from pathlib import Path
import harp
import pandas as pd
import numpy as np
import xarray as xr
import yaml

from collections.abc import Mapping

from Refactor.HarpExtender.harp_extender_core import construct_device_reader, collect_device_folders, collect_registers, collect_device_dfs
from Refactor.Timestamps.timestamps import collect_timestamps, collect_timestamps_dict, collect_timestamps_nested_dict
from Refactor.LookupArrays.lookup_arrays import construct_base_da, construct_lookup_array, ulookup, LookupAccessorConstructor, CDLAccessor, update_channel_type_da

################################################################################





################################################################################
# Base MultiDevice Class
################################################################################

class MultiDevice:
     '''
     Class to manage loading and processing of Multi-Device data from HARP devices.
     -----------------------------------
     

     '''
     def __init__(self,
                  experiment_directory_path: str | Path = None,
                  harp_device_yaml_path: str | Path = None,
                  device_type: str = None,
                  device_list: list | None = None,
                  device_IDs: dict | None = None,
                  device_registers_dict: dict | None = None,
                  unified_dims: list | None = None,
                  data_localIDs: dict | list | None = None,
                  data_registerIDs: dict | list | None = None, 
                  data_keys: list | None = None,
                  test_values: bool = False,
                  fill_value = None,
                  align: str = 'exact',
                  verbose: bool = False,
                  only_errors: bool = False,
                  validate_devices: bool = False,
                  validate_deviceIDs: bool = False,
                  validate_device_registers: bool = False,
                  validate_register_channels: bool = False,
                  fill_values: dict | None = None,
                  data_da_attributes: dict | None = None,
                  data_lookup_attributes: dict | None = None,
                  Return: bool = False
                  ):

            super(MultiDevice, self).__init__()
            
            ##===== User-Defined Parameters for Data Structure =================
            self.experiment_directory_path= experiment_directory_path
            self.harp_device_yaml_path= harp_device_yaml_path
            self.device_type= device_type
            self.device_list= device_list
            self.device_IDs= device_IDs
            self.device_registers_dict= device_registers_dict
            self.unified_dims= unified_dims
            self.data_localIDs= data_localIDs
            self.data_registerIDs= data_registerIDs
            self.data_keys= data_keys
            
            ##===== User-Defined Parameters for Processing =====================
            self.test_values= test_values
            self.fill_value= fill_value
            self.align= align
            self.verbose= verbose
            self.only_errors= only_errors
            self.validate_devices= validate_devices
            self.validate_deviceIDs= validate_deviceIDs
            self.validate_device_registers= validate_device_registers
            self.validate_register_channels= validate_register_channels
            self.fill_values= fill_values
            self.data_da_attributes= data_da_attributes
            self.data_lookup_attributes= data_lookup_attributes
            self.Return= Return

            ##===== Setup ======================================================

            self.harp_device_reader= self.construct_device_reader()
            self.device_folders= self.collect_device_folders(verbose= self.verbose)

            self.registers= self.collect_registers(verbose= self.verbose)
            self.device_dfs_dict= self.collect_device_dfs_dicts(verbose= self.verbose)





     def construct_device_reader(self,
                                ):
        '''
        Wrapper for construct_device_reader function from harp_extender_core.
        '''
        reader= construct_device_reader(harp_device_yaml_path= self.harp_device_yaml_path)
        return reader

     def collect_device_folders(self, verbose= False):
        '''
        Wrapper for collect_device_folders function from harp_extender_core.
        '''
        device_folders= collect_device_folders(experiment_directory_path= self.experiment_directory_path,
                                              device_type= self.device_type,
                                              verbose= verbose
                                              )
        return device_folders

     def collect_registers(self,
                          verbose= False
                          ):
        '''
        Wrapper for collect_registers function from harp_extender_core.
        '''
        registers= collect_registers(harp_device_yaml_path= self.harp_device_yaml_path,
                                     device_list= self.device_list,
                                     device_IDs= self.device_IDs,
                                     device_registers_dict= self.device_registers_dict,
                                     verbose= verbose
                                     )
        return registers

    
     def collect_device_dfs_dicts(self,
                                   verbose= False
                                   ):
        '''
        Wrapper for collect_device_dfs function from harp_extender_core.
        '''
        device_dfs_dict= collect_device_dfs(experiment_directory_path= self.experiment_directory_path,
                                           harp_device_yaml_path= self.harp_device_yaml_path,
                                           device_type= self.device_type,
                                           device_list= self.device_list,
                                           device_IDs= self.device_IDs,
                                           device_registers_dict= self.device_registers_dict,
                                           verbose= verbose
                                           )
        return device_dfs_dict
     
     def construct_base_da(self,
                           values= None,
                           unified_dim= None,
                           virtual_coord_name= None,
                           virtual_coord_names= None,
                           virtual_map_dict= None,
                           dict_of= 'tuples',
                           times= None,
                           name= None,
                           verbose= False,
                           ):
        '''
        Wrapper for construct_base_da function from lookup_arrays.
        '''
        base_da= construct_base_da(values= values,
                                   unified_dim= unified_dim,
                                   virtual_coord_names= virtual_coord_names,
                                   virtual_map_dict= virtual_map_dict,
                                   dict_of= dict_of,
                                   times= times,
                                   name= name,
                                   test_help_values= self.test_values,
                                   verbose= verbose
                                   )
        return base_da
     
     def construct_lookup_array(self,
                                unified_dim= None,
                                unified_coord_name= None,
                                virtual_coord_names= None,
                                virtual_map_dict= None,
                                dict_of= 'tuples',
                                name= None,
                                verbose= False,
                                ):
        '''
        Wrapper for construct_lookup_array function from lookup_arrays.
        '''
        lookup_array= construct_lookup_array(unified_dim= unified_dim,
                                             unified_coord_name= unified_coord_name,
                                             virtual_coord_names= virtual_coord_names,
                                                virtual_map_dict= virtual_map_dict,
                                                dict_of= dict_of,
                                                name= name,
                                                verbose= verbose
                                                )
        return lookup_array
     
     def construct_virtual_map_dict(self):
          '''
          Placeholder for construct_virtual_map_dict method to be implemented in subclasses.
          '''
          raise NotImplementedError("Subclasses must implement construct_virtual_map_dict method.")
     



     def update_base_da(
        data_array: xr.DataArray = None,
        virtual_map_dict: dict = None,
        device_dfs_dict: dict = None,
        virtual_key_map: dict = None,   # {source_key: {localID: reg_address}}
        source_key: str = None,
        fill_value=None,
        verbose: bool = False,
    ):
        '''
        Wrapper for update_channel_type_da function from lookup_arrays. Use update_channel_type_da instead.
        '''
        updated_da= update_channel_type_da(channel_type_da= data_array,
                                           channel_ref_dict= virtual_map_dict,
                                             dict_of_devices_registers_dfs_dicts= device_dfs_dict,
                                             device_register_dfs_dict= None,
                                             channel_type_registerIDs= virtual_key_map,
                                             type_key= source_key,
                                             fill_value= fill_value,
                                             verbose= verbose
        )
        return updated_da
     
    #  def construct_data_outputs(self,
    #                             update: bool= True,
    #                             use_datakey: str | None= None,
    #                             fill_value= None,
    #                             verbose: bool= False,
    #                             data_assign_attrs: dict | None= None,
    #                             lookup_assign_attrs: dict | None= None,
    #                             ):
    #     '''
    #     Get all data outputs (data DataArray, lookup DataArray, virtual map dict) for a given data key. Requires subclass implementation of construct_virtual_map_dict method.
    #     -----------------------------------
    #     Parameters:
    #         update (bool): If True, updates the data DataArray using the device data.
    #         use_datakey (str): The data key to process.
    #         fill_value: Value to use for missing data when updating data DataArrays.
    #         verbose (bool): If True, prints detailed output during processing.
    #         data_assign_attrs (dict): Attributes to assign to the data DataArray.
    #         lookup_assign_attrs (dict): Attributes to assign to the lookup DataArray.
    #     Returns:
    #         data_da (xr.DataArray): The constructed or updated data DataArray.
    #         lookup_da (xr.DataArray): The constructed lookup DataArray.
    #         virtual_map_dict (dict): The constructed virtual map dictionary.
    #     '''
        
    #     #===== Set Default Parameters 
    #     if verbose is None: verbose= self.verbose
    #     if fill_value is None: fill_value= self.fill_value
        
    #     if use_datakey is None: raise ValueError("use_datakey must be provided.")

    #     if use_datakey not in self.data_keys:   raise ValueError(f"use_datakey '{use_datakey}' not found in data_keys.")

    #     if data_assign_attrs is not None:
    #         if verbose: print(f'Custom data_assign_attrs not supported in base MultiDevice class, proceeding')
    #     if lookup_assign_attrs is not None:
    #         if verbose: print(f'Custom lookup_assign_attrs not supported in base MultiDevice class, proceeding')
        
    #     #===== Construct Virtual Map Dict 
    #     virtual_map_dict= self.construct_virtual_map_dict(data_key= use_datakey)
    #     if verbose: print(f'Constructed virtual_map_dict for data_key "{use_datakey}" with {len(virtual_map_dict)} entries.')

    #     #===== Construct Base DataArray
    #     data_array= self.construct_base_da(unif)

     def construct_data_outputs(
        self,
        use_datakey: str,
        update: bool = True,
        fill_value=None,
        verbose: bool | None = None,
        name: str | None = None,
        lookup_name: str | None = None,
        ):
        if verbose is None:
            verbose = self.verbose

        if use_datakey not in self.data_keys:
            raise ValueError(f"use_datakey '{use_datakey}' not in data_keys {self.data_keys}")

        if fill_value is None:
            fill_value = self.fill_value

        # 1) subclass mapping: {unified_label: (device, localID)}
        virtual_map_dict = self.construct_virtual_map_dict(data_key=use_datakey)
        if not isinstance(virtual_map_dict, dict) or not virtual_map_dict:
            raise ValueError("construct_virtual_map_dict must return a non-empty dict.")

        unified_labels = list(virtual_map_dict.keys())

        # 2) timestamps union for registers involved in this data_key (INLINE, no helper method)
        reg_map = self.data_registerIDs[use_datakey]              # {localID: reg_addr}
        needed_regs = sorted(set(reg_map.values()))

        times = set()
        for device in self.device_list:
            dev_regs = self.device_dfs_dict.get(device, {})
            for reg in needed_regs:
                df = dev_regs.get(reg, None)
                if df is None:
                    continue
                times.update(df.index)

        times = sorted(times)
        if not times:
            raise ValueError(f"No timestamps found for data_key '{use_datakey}' (check device_dfs_dict/registers).")

        # 3) construct base + lookup
        data_name = name or f"{use_datakey}_DataArray"
        data_da = self.construct_base_da(
            values=None,
            unified_dim=unified_labels,
            virtual_coord_names=self.virtual_coord_names,
            virtual_map_dict=virtual_map_dict,
            times=times,
            name=data_name,
            verbose=verbose,
        )

        lkp_name = lookup_name or f"{use_datakey}_Lookup"
        lookup_da = self.construct_lookup_array(
            unified_dim=unified_labels,
            unified_coord_name=self.unified_coord_name,
            virtual_coord_names=self.virtual_coord_names,
            virtual_map_dict=virtual_map_dict,
            name=lkp_name,
            verbose=verbose,
        )

        # 4) update from device dfs
        if update:
            data_da = self.update_base_da(
                data_array=data_da,
                virtual_map_dict=virtual_map_dict,
                source_key=use_datakey,
                fill_value=fill_value,
                verbose=verbose,
            )

        # cache
        self.data_das[use_datakey] = data_da
        self.lookup_das[use_datakey] = lookup_da
        self.virtual_maps[use_datakey] = virtual_map_dict

        return data_da, lookup_da, virtual_map_dict

     





################################################################################

    #  def update_base_da(
    #     data_array: xr.DataArray = None,
    #     virtual_map_dict: dict = None,
    #     device_dfs_dict: dict = None,
    #     virtual_key_map: dict = None,   # {source_key: {localID: reg_address}}
    #     source_key: str = None,
    #     fill_value=None,
    #     verbose: bool = False,
    # ):
    #     '''
    #     Wrapper for update_channel_type_da function from lookup_arrays. Use update_channel_type_da instead.
    #     '''
    #     updated_da= update_channel_type_da(channel_type_da= None,
    #                                        channel_ref_dict= None,
    #                                          dict_of_devices_registers_dfs_dicts= None,
    #                                          device_register_dfs_dict= None,
    #                                          channel_type_registerIDs= None,
    #                                          type_key= None,
    #                                          fill_value= None,
    #                                          verbose= verbose
    #     )
    #     return updated_da


################################################################################
#   Nosepoke Class 
################################################################################

class BehaviorNosepoke(MultiDevice):
    def __init__(
        self,
        experiment_directory_path=None,
        harp_device_yaml_path=None,
        device_type="Behavior",
        device_list=None,
        device_IDs=None,
        device_registers_dict=None,

        # nosepoke-specific
        channel_list=None,                 # list[str] length = len(device_list)*len(localIDs_per_device)
        channel_type_localIDs=None,        # dict[data_key -> list[localID]]
        channel_type_registerIDs=None,     # dict[data_key -> dict[localID -> reg_addr]]
        type_keys=None,                    # list[str] == data_keys

        test_values=False,
        fill_value=None,
        verbose=False,
    ):
        self.channel_list = channel_list
        self.channel_type_localIDs = channel_type_localIDs
        self.type_keys = type_keys

        super().__init__(
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            data_keys=type_keys,
            data_registerIDs=channel_type_registerIDs,
            unified_coord_name="channel",
            virtual_coord_names=("device", "localID"),
            test_values=test_values,
            fill_value=fill_value,
            verbose=verbose,
        )

        # hard requirements for this subclass
        if not isinstance(self.channel_type_localIDs, dict) or not self.channel_type_localIDs:
            raise ValueError("channel_type_localIDs must be {type_key: [localIDs]}")

        if self.channel_list is None:
            # you can remove this if you want channel_list ALWAYS explicit
            n = len(self.device_list) * len(self.channel_type_localIDs[self.data_keys[0]])
            self.channel_list = [f"Nosepoke{i}" for i in range(n)]

    def construct_virtual_map_dict(self, data_key: str) -> dict:
        localIDs = self.channel_type_localIDs[data_key]
        expected = len(self.device_list) * len(localIDs)

        if len(self.channel_list) != expected:
            raise ValueError(
                f"channel_list len={len(self.channel_list)} but expected {expected} "
                f"(devices={len(self.device_list)} x localIDs={len(localIDs)})"
            )

        vmap = {}
        i = 0
        for device in self.device_list:
            for localID in localIDs:
                vmap[self.channel_list[i]] = (device, localID)
                i += 1
        return vmap

###################################

# class BehaviorNosepoke:
    ''' 
    Class to manage loading and processing of Behavior Nosepoke data from HARP devices.
    -----------------------------------

    Attributes:
        experimental_directory_path (str or Path): Path to experimental directory.
        harp_device_yaml_path (str or Path): Path to harp_device yaml file.
        device_type (str): Device type to load. Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc.
        device_list (list of str): List of device names to load.
        device_IDs (dict of str: str): Dict of device names to IDs.
        device_registers_dict (dict of str: list of str): Dict of device names to list of register addresses.
        channel_list (list of str): List of channel names to load.
        channel_type_localIDs (dict of str: list of str): Dict of type_key to list of local IDs.
        channel_type_registerIDs (dict of str: dict of str: str): Dict of type_key to dict of localID to register address.
        type_keys (list of str): List of type keys to load.
        test_help_values (bool): If True, use test helper values instead of NaNs for channel type DataArrays.
        fill_value: Value to use for missing data when updating channel type DataArrays.
        align (str): Alignment mode for updating DataArrays. Kept for compatibility, currently only 'exact' is supported.
        verbose (bool): If True, prints detailed output during processing.
        only_errors (bool): If True, only prints errors during processing.
        validate_devices (bool): If True, runs device validation checks.
        validate_deviceIDs (bool): If True, runs device ID validation checks.
        validate_device_registers (bool): If True, runs device register validation checks.
        validate_register_channels (bool): If True, runs register channel validation checks.
        fill_values (dict of str: value): Dict of type_key to fill_value.
        typekey_da_attributes (dict of str: dict): Dict of type_key to dict of attributes.
        typekey_lookup_attributes (dict of str: dict): Dict of type_key to dict of attributes.
        Return: If true, then additional information is returned.
    -------------
    


    
    '''

    def __init__(self,
                 experimental_directory_path= None,         # Path to experimental directory (str or Path)
                 harp_device_yaml_path= None,               # Path to harp_device yaml file (str or Path)
                 device_type= 'Behavior',                   # Device type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc.
                 device_list= None,                         # List of device names to load (list of str)
                 device_IDs= None,                          # Dict of device names to IDs (dict of str: str)
                 device_registers_dict= None,               # Dict of device names to list of register addresses (dict of str: list of str). dict= {f'{device_list[n]}': [registers],...}
                 channel_list= None,                        # List of channel names to load (list of str)
                 channel_type_localIDs= None,               # Dict of type_key to list of local IDs (dict of str: list of str). E.g. {'Activations': ['DIPort0', 'DIPort1',...],...} ?
                 channel_type_registerIDs= None,            # Dict of type_key to dict of localID to register address (dict of str: dict of str: str). E.g. {'Activations': {'DIPort0': '32', 'DIPort1': '32',...},...} ?
                 type_keys= None,                                   # List of type keys to load (list of str). Obtained from list(channel_type_localIDs.keys())
                 selected_type_key= None,                     # Type key to load (str). Must be one of type_keys.
                 channel_ref_dict= None,                    # Dict of channel names to (device, localID) tuples (dict of str: (str, str)). E.g. {'Nosepoke0': ('Behavior0', 'DIPort0'),...}
                 test_help_values= False,                   # If True, use test helper values instead of NaNs for channel type DataArrays (bool)
                 fill_value= None,                          # Value to use for missing data when updating channel type DataArrays (e.g., 0, False). If None, will use NaN for floats and False for bools.
                 align= 'exact',                            # Alignment mode for updating DataArrays (str). Kept for compatibility, currently only 'exact' is supported.
                 verbose= False,                            # If True, prints detailed output during processing (bool)
                 only_errors= False,                             # If True, only prints errors during processing (bool)
                 validate_devices= False,                   # If True, runs device validation checks (bool)
                 validate_deviceIDs= False,                 # If True, runs device ID validation checks (bool)
                 validate_device_registers= False,          # If True, runs device register validation checks (bool)
                 validate_register_channels= False,         # If True, runs register channel validation checks (bool)
                 fill_values= None,                          # Dict of type_key to fill_value (dict of str: value).If None, will use default fill values. E.g. {'Activations': False, 'Counts': 0, ...}
                 typekey_da_attributes= None,                 # Dict of type_key to dict of attributes (dict of str: dict). If None, then will use default attributes. E.g. {'Activations': {'description': f'DataArray for channels of type {use_typekey}','source': 'Constructed using get_all_channel_type_outputs function'}}
                 typekey_lookup_attributes= None,              # Dict of type_key to dict of attributes (dict of str: dict). If None, then will use default attributes. E.g. {'Activations': {'description': 'Channel lookup matrix indicating presence of channels across devices and local IDs','source': 'Constructed using get_all_channel_type_outputs function'}}
                 Return= False
                 
                 ):
        
        super(BehaviorNosepoke, self).__init__()

        ##===== User-Defined Parameters for Data Structure =================
        self.experiment_directory_path= experimental_directory_path
        self.harp_device_yaml_path= harp_device_yaml_path
        
        self.device_type= device_type
        
        self.device_list= device_list
        self.device_IDs= device_IDs
        self.device_registers_dict= device_registers_dict
        
        self.channel_list= channel_list
        self.channel_type_localIDs= channel_type_localIDs
        self.channel_type_registerIDs= channel_type_registerIDs
        
        self.type_keys= type_keys
        # self.type_key= selected_type_key
        #self.channel_ref_dict= channel_ref_dict

        ##===== User-Defined Parameters for Processing =====================
        self.test_help_values= test_help_values
        self.fill_value= fill_value
        self.align= align
        self.verbose= verbose
        self.only_errors= only_errors
        self.validate_devices= validate_devices
        self.validate_deviceIDs= validate_deviceIDs
        self.validate_device_registers= validate_device_registers
        self.validate_register_channels= validate_register_channels

        self.fill_values= fill_values
        self.typekey_da_attributes= typekey_da_attributes
        self.typekey_lookup_attributes= typekey_lookup_attributes
        self.Return= Return

        ##===== Setup ======================================================

        self.harp_device_reader= create_device_reader(harp_device_yaml_path=self.harp_device_yaml_path)

        self.device_folders= get_device_folders(experiment_directory_path=self.experiment_directory_path, 
                                                device_type=self.device_type
                                                )
        
        self.device_subfolders= get_device_subfolders(experiment_directory_path=self.experiment_directory_path,
                                                    device_type=self.device_type,
                                                       )
        
        self.dict_of_devices_register_dfs_dicts= get_dict_of_devices_register_dfs_dicts(experiment_directory_path= self.experiment_directory_path,
                                                                                        harp_device_yaml_path= self.harp_device_yaml_path,
                                                                                        device_type= self.device_type,
                                                                                        device_list= self.device_list,
                                                                                        device_IDs= self.device_IDs,
                                                                                        device_registers_dict= self.device_registers_dict,
                                                                                        Return= self.Return,
                                                                                        verbose= self.verbose,
                                                                                        )
        self.dict_of_devices_registers_dfs_dicts = self.dict_of_devices_register_dfs_dicts
        self.channel_data_arrays= {}  # To store channel type DataArrays
        self.channel_lookup_data_arrays= {}  # To store channel lookup DataArrays
        self.channel_reference_dicts= {}  # To store channel reference dictionaries


        for type_key in self.type_keys:
            if self.verbose:
                print(f'\nProcessing type_key: {type_key}')

            #===== Determine Fill Value =====
            if self.fill_values is not None and type_key in self.fill_values:
                tk_fill_value= self.fill_values[type_key]
            else:
                tk_fill_value= self.fill_value
            
            #===== Determine TypeKey DataArray Attributes =====
            if self.typekey_da_attributes is not None and type_key in self.typekey_da_attributes:
                tk_ctda_attrs= self.typekey_da_attributes[type_key]
            else:
                tk_ctda_attrs= None
            #===== Determine TypeKey Lookup Attributes =====
            if self.typekey_lookup_attributes is not None and type_key in self.typekey_lookup_attributes:
                tk_lookup_attrs= self.typekey_lookup_attributes[type_key]
            else:
                tk_lookup_attrs= None
            

            #===== Use Main Function to Get ctda, clda, crd =====
            ctda, clda, crd= self.get_channel_type_outputs(
                                                        #   update= True,
                                                          use_typekey= type_key,
                                                          fill_value= tk_fill_value,
                                                          verbose= self.verbose,
                                                          ctda_assign_attrs= tk_ctda_attrs,
                                                          lookup_assign_attrs= tk_lookup_attrs
                                                          )
            self.channel_data_arrays[type_key]= ctda
            self.channel_lookup_data_arrays[type_key]= clda
            self.channel_reference_dicts[type_key]= crd



        ##===== Methods That Need to be Called Multiple Times ======================================================================================

        ##===== Channel Reference Dict & Associated Mini-Functions =========

    def create_channel_ref_dict(self,
                                    type_key=None,
                                    ):
            '''
            Create a channel reference dictionary mapping channel names to (device, localID) tuples.
            -----------------------------------
            Parameters:
                self.channel_list (list): List of channel names.
                self.device_list (list): List of device names.
                self.channel_type_localIDs (dict): Dictionary mapping type keys to lists of local IDs. Requires type_key to be specified.
            
            Arguments:
                type_key (str): The type_key corresponding to the channel_type_da. Must be one of self.type_keys.
            
            Returns:
                channel_ref_dict (dict): Dictionary mapping channel names to (device, localID) tuples.
            '''

            if type_key is None:
                raise ValueError("type_key must be provided to create channel_ref_dict.")
            
            if type_key not in self.channel_type_localIDs:
                raise ValueError(f"type_key '{type_key}' not found in channel_type_localIDs.")
        


            channel_list= self.channel_list
            channel_type_localIDs= self.channel_type_localIDs
            device_list= self.device_list

            channel_ref_dict= {
                f'{channel_list[0]}':  (f'{device_list[0]}', f'{channel_type_localIDs[type_key][0]}'),
                f'{channel_list[1]}':  (f'{device_list[0]}', f'{channel_type_localIDs[type_key][1]}'),
                f'{channel_list[2]}':  (f'{device_list[0]}', f'{channel_type_localIDs[type_key][2]}'),

                f'{channel_list[3]}':  (f'{device_list[1]}', f'{channel_type_localIDs[type_key][0]}'),
                f'{channel_list[4]}':  (f'{device_list[1]}', f'{channel_type_localIDs[type_key][1]}'),
                f'{channel_list[5]}':  (f'{device_list[1]}', f'{channel_type_localIDs[type_key][2]}'),

                f'{channel_list[6]}':  (f'{device_list[2]}', f'{channel_type_localIDs[type_key][0]}'),
                f'{channel_list[7]}':  (f'{device_list[2]}', f'{channel_type_localIDs[type_key][1]}'),
                f'{channel_list[8]}':  (f'{device_list[2]}', f'{channel_type_localIDs[type_key][2]}'),

                f'{channel_list[9]}':  (f'{device_list[3]}', f'{channel_type_localIDs[type_key][0]}'),
                f'{channel_list[10]}': (f'{device_list[3]}', f'{channel_type_localIDs[type_key][1]}'),
                f'{channel_list[11]}': (f'{device_list[3]}', f'{channel_type_localIDs[type_key][2]}'),

                f'{channel_list[12]}': (f'{device_list[4]}', f'{channel_type_localIDs[type_key][0]}'),
                f'{channel_list[13]}': (f'{device_list[4]}', f'{channel_type_localIDs[type_key][1]}'),
                f'{channel_list[14]}': (f'{device_list[4]}', f'{channel_type_localIDs[type_key][2]}'),

                f'{channel_list[15]}': (f'{device_list[5]}', f'{channel_type_localIDs[type_key][0]}'),
                f'{channel_list[16]}': (f'{device_list[5]}', f'{channel_type_localIDs[type_key][1]}'),
                f'{channel_list[17]}': (f'{device_list[5]}', f'{channel_type_localIDs[type_key][2]}'),
            }

            return channel_ref_dict
        
    def get_channel_type_da_name(self,
                                     type_key=None,
                                     ):
            '''
            Generate a standardised name for the channel type DataArray based on the type_key.

            Args:
                type_key (str): The type_key corresponding to the channel_type_da. Must be one of self.type_keys.

            Returns:
                str: The standardised name for the channel type DataArray.
            '''

            if type_key is None:
                raise ValueError("type_key must be provided.")

            if type_key not in self.type_keys:
                raise ValueError(f"type_key '{type_key}' is not valid.")
            
            channel_type_da_name= f'{type_key}_DataArray'

            return channel_type_da_name
        
    def get_channels_direct_from_ref_dict(self,
                                          type_key=None
                                          ):
            '''
            Get the list of channels associated with a given type_key by cross-referencing the channel_ref_dict. This is not strictly necessary since channel_list is already defined, but it provides a way to verify consistency.

            Args:
                type_key (str): The type_key corresponding to the channel_type_da. Must be one of self.type_keys.

            Returns:
                list: List of channel names.
            '''

            if type_key is None:
                raise ValueError("type_key must be provided.")

            if type_key not in self.type_keys:
                raise ValueError(f"type_key '{type_key}' is not valid.")
            
            channel_ref_dict= self.create_channel_ref_dict(type_key= type_key)

            channels= list(channel_ref_dict.keys())

            return channels
            



        ##===== Type Key Timestamps ========================================

        # 6a) Get DF Timestamps

    def get_df_timestamps(df):
            """
            Extract timestamps from a DataFrame's index.

            Parameters:
                df (pd.DataFrame): The DataFrame from which to extract timestamps.

            Returns:
                pd.DatetimeIndex: The timestamps extracted from the DataFrame's index.
            """
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame.")
            
            if df.index.name != 'Time':
                raise ValueError("DataFrame index must be named 'Time'.")

            return df.index
        
        # 6b) Get DF Dict Timestamps

    def get_df_dict_timestamps(self,
                               df_dict= None, 
                                   return_type='list'
                                   ):
            """
            Extract timestamps from each DataFrame in a dictionary.

            Parameters:
                df_dict (dict): A dictionary where keys are identifiers and values are pandas DataFrames.
                return_type (str, optional): If 'list', returns a sorted list of all timestamps. If 'dict', returns a dictionary of timestamps per DataFrame. Default is 'list'. If both are needed, call the function twice.

            Returns:
                list of all timestamps from each DataFrame in the dictionary, sorted in ascending order.
                dict where keys are the same as df_dict and values are the timestamps from each DataFrame.
            """
            if not isinstance(df_dict, dict):
                raise ValueError("Input must be a dictionary.")
            
            if return_type is None:
                raise ValueError("return_type must be specified as 'list' or 'dict'.")
            
            if return_type == 'list':
                all_timestamps = []
                for key, df in df_dict.items():
                    if not isinstance(df, pd.DataFrame):
                        raise ValueError(f"Value for key '{key}' is not a pandas DataFrame.")
                    if df.index.name != 'Time':
                        raise ValueError(f"DataFrame for key '{key}' must have index named 'Time'.")
                    
                    all_timestamps.extend(df.index)

                return sorted(set(all_timestamps))
            
            elif return_type == 'dict':
                timestamps_dict = {}
                for key, df in df_dict.items():
                    if not isinstance(df, pd.DataFrame):
                        raise ValueError(f"Value for key '{key}' is not a pandas DataFrame.")
                    if df.index.name != 'Time':
                        raise ValueError(f"DataFrame for key '{key}' must have index named 'Time'.")
                    
                    timestamps_dict[key] = df.index

                return timestamps_dict
            
            elif return_type == 'both':
                all_timestamps = []
                timestamps_dict = {}
                for key, df in df_dict.items():
                    if not isinstance(df, pd.DataFrame):
                        raise ValueError(f"Value for key '{key}' is not a pandas DataFrame.")
                    if df.index.name != 'Time':
                        raise ValueError(f"DataFrame for key '{key}' must have index named 'Time'.")

                    all_timestamps.extend(df.index)
                    timestamps_dict[key] = df.index

                return sorted(set(all_timestamps)), timestamps_dict
            
        # 6c) Get DF Dict Timestamps using Nested Dict

    def get_df_dict_timestamps_from_nested_dict(self,
                                                    #dict_of_devices_registers_dfs_dicts=None,
                                                    type_key=None,
                                                    #devices=None,
                                                    return_type='list',
                                                    verbose=False
            ):
                """
                Collect timestamps across all devices/registers relevant to the given type_key from the nested dict_of_devices_registers_dfs_dicts using get_df_dict_timestamps.

                Parameters:
                    dict_of_devices_registers_dfs_dicts (dict): dict_of_devices_registers_dfs_dicts {device: {register_address: DataFrame(Time index, localID columns)}}
                    type_key (str): channel type to use (e.g., the global type_key)
                    devices (list or None): optional subset of devices to include; defaults to all in dict_of_devices_registers_dfs_dicts
                    return_type (str): 'list', 'dict', or 'both' (passed through to get_df_dict_timestamps)
                    verbose (bool): whether to print verbose output

                Returns:
                    list | dict | (list, dict): timestamps per return_type
                """
                dict_of_devices_registers_dfs_dicts= self.dict_of_devices_registers_dfs_dicts
                devices= self.device_list

                if not isinstance(dict_of_devices_registers_dfs_dicts, dict) or not dict_of_devices_registers_dfs_dicts:
                        print("Warning: dict_of_devices_registers_dfs_dicts is empty or not a dict.")
                        return [] if return_type == 'list' else ({} if return_type == 'dict' else ([], {}))

                # Use provided devices or all devices present in the nested dict
                devices = devices if devices is not None else list(dict_of_devices_registers_dfs_dicts.keys())

                # Map localID -> register address for this type_key, then get unique register addresses
                reg_map = self.channel_type_registerIDs[type_key]  # e.g., {'DIPort0': '32', ...}
                needed_registers = sorted(set(reg_map.values()))

                # Build a flat dict of DataFrames to feed into get_df_dict_timestamps
                # Keyed by "{device}:{register}"
                df_dict = {}
                for device in devices:
                        if device not in dict_of_devices_registers_dfs_dicts:
                                print(f"Warning: Device {device} not found in dict_of_devices_registers_dfs_dicts; skipping.")
                                continue

                        dev_regs = dict_of_devices_registers_dfs_dicts[device]
                        for reg in needed_registers:
                                if reg not in dev_regs:
                                        print(f"Warning: Device {device} missing register {reg}; skipping.")
                                        continue

                                # Restrict columns to localIDs relevant to this type_key and register
                                localIDs_for_reg = [lid for lid, r in reg_map.items() if r == reg and lid in dev_regs[reg].columns]
                                if not localIDs_for_reg:
                                        print(f"Warning: Device {device} register {reg} has no expected localIDs present; skipping.")
                                        continue

                                df_subset = dev_regs[reg][localIDs_for_reg]
                                df_dict[f"{device}:{reg}"] = df_subset

                if not df_dict:
                        print("Warning: No DataFrames collected for the specified type_key.")
                        return [] if return_type == 'list' else ({} if return_type == 'dict' else ([], {}))

                if return_type == 'list':
                        return self.get_df_dict_timestamps(df_dict=df_dict, return_type='list')
                elif return_type == 'dict':
                        return self.get_df_dict_timestamps(df_dict=df_dict, return_type='dict')
                elif return_type == 'both':
                        return self.get_df_dict_timestamps(df_dict=df_dict, return_type='both')
                else:
                        raise ValueError("return_type must be one of 'list', 'dict', or 'both'.")
        


        ##===== Channel Type Data Array ====================================
        
        # 7) Construct Channel Type DataArray

    def construct_channel_type_da(self,
                                      values= None, 
                                      #test_help_values= False, 
                                      use_typekey= None,
                                      channels=None, 
                                      channels_ref_dict= None, 
                                      times= None, 
                                      assign_attrs= None,
                                      #name='channel_type_da'
                                      name= None,
                                      verbose= None
                                      ):
            '''
            Construct a Placeholder DataArray for a set of given channels for all timepoints across all devices for a specific register type.
            -----------------------------------

            Parameters:
                values (list or np.ndarray): Values to populate the DataArray. If None, a placeholder array of NaNs will be created.
                test_help_values (bool): If True, use test helper values instead of the main values.
                use_typekey(str or None): If provided;
                    - channel_ref_dict will be created using this type_key via create_channel_ref_dict, unless channels_ref_dict is provided directly.
                    - name will be set using get_channel_type_da_name, unless name is provided directly.
                    - channels will be set using self.channel_list, unless channels is provided directly.
                    - times will be set using get_df_dict_timestamps_from_nested_dict with return_type='list', unless times is provided directly.
                    - If None, channels, channels_ref_dict, and name must be provided directly.
                (i) channels (list): List of channel names to include in the DataArray.
                (i) channels_ref_dict (dict): Dictionary mapping channel names to their reference information.
                (i) times (list): List of timepoints to include in the DataArray. This should be obtained using get_df_dict_timestamps with return_type='list'.
                assign_attrs (dict or None): Optional dictionary of attributes to assign to the DataArray. If None, default attributes will be used.
                (i) name (str): Name of the DataArray.
                (ii) verbose (bool or None): If True, prints detailed output during processing. If None, defaults to self.verbose.

            Notes:
                i. Optional. If provided, will overwrite values obtained from use_typekey functionality.
                ii. Optional. If None, will be obtained from self.verbose

            Returns:
                A DataArray populated with the specified values, or a placeholder array if no values are provided.
            '''

            test_help_values= self.test_help_values

            
            #--------------------------------------------------------------------------------------------------------------------
            if verbose is None:
                verbose= self.verbose



            if use_typekey is not None:
                if use_typekey not in self.type_keys:
                    raise ValueError(f"use_typekey '{use_typekey}' is not valid.")

                #---- Get channels ----
                if channels is None:
                    channels= self.channel_list
                elif channels is not None and verbose:
                    # print("Warning: channels provided directly will override those obtained from self.channel_list.")
                    # See if provided channels match those which would be obtained from self.channel_list
                    if set(channels) != set(self.channel_list):
                        print("Warning: Provided channels do not match those from self.channel_list.")

                #---- Get channels_ref_dict ----
                if channels_ref_dict is None:
                    channels_ref_dict= self.create_channel_ref_dict(type_key= use_typekey)
                elif channels_ref_dict is not None and verbose:
                    # print("Warning: channels_ref_dict provided directly will override that obtained from create_channel_ref_dict.")
                    # See if provided channels_ref_dict match those which would be obtained from create_channel_ref_dict
                    if set(channels_ref_dict.keys()) != set(self.create_channel_ref_dict(type_key=use_typekey).keys()):
                        print("Warning: Provided channels_ref_dict do not match those from create_channel_ref_dict.")

                #---- Get times ----
                if times is None:
                    times= self.get_df_dict_timestamps_from_nested_dict(type_key= use_typekey,
                                                                        return_type='list'
                                                                        )
                elif times is not None and verbose:
                    # print("Warning: times provided directly will override those obtained from get_df_dict_timestamps_from_nested_dict.")
                    # See if provided times match those which would be obtained from get_df_dict_timestamps_from_nested_dict
                    if set(times) != set(self.get_df_dict_timestamps_from_nested_dict(type_key=use_typekey, return_type='list')):
                        print("Warning: Provided times do not match those from get_df_dict_timestamps_from_nested_dict.")

                #---- Get name ----
                if name is None:
                    name= self.get_channel_type_da_name(type_key= use_typekey)
                elif name is not None and verbose:
                    # print("Warning: name provided directly will override that obtained from get_channel_type_da_name.")
                    # See if provided name matches that which would be obtained from get_channel_type_da_name
                    if name != self.get_channel_type_da_name(type_key=use_typekey):
                        print("Warning: Provided name does not match that from get_channel_type_da_name.")
                
            if assign_attrs is None:
                assign_attrs = {
                    'description': f'DataArray for channels of type {name}',
                    'source': 'Constructed using construct_channel_type_da function'
                }
                if verbose:
                    print("No assign_attrs provided; using default attributes: ", assign_attrs)
            #--------------------------------------------------------------------------------------------------------------------

            assert times is not None and channels is not None and channels_ref_dict is not None, "times, channels, and channels_ref_dict must be provided."

            if values is None and test_help_values is False:
                if verbose:
                    print(f'No values provided, and test_help_values is False. Creating zeros placeholder DataArray with {len(times)} timestamps and {len(channels)} channels.')
                values= np.zeros((len(times), len(channels)),dtype= bool
                                )
            
            if values is None and test_help_values is True:
                print(f'No values provided, but test_help_values is True. Creating test helper DataArray with {len(times)} timestamps and {len(channels)} channels.')
                value_list= []
                for time in times:
                    time_values= []
                    for channel in channels:
                        device, localID = channels_ref_dict[channel]
                        time_values.append(f'({channel}, {device}, {localID})')
                    value_list.append(time_values)
                values= np.array(value_list)
                if verbose:
                    print(f'Constructed test helper values array with shape: {values.shape}')
            # if values is not None and test_help_values is True:
            #     raise ValueError('Cannot provide both values and set test_help_values to True. Choose one or the other.')

            devices= [channels_ref_dict[channel][0] for channel in channels]
            localIDs= [channels_ref_dict[channel][1] for channel in channels]

            da= xr.DataArray(
                data=values,
                dims= ['Time', 'channel'],
                coords= {
                    'Time': times,
                    'channel': channels,
                    'device': ('channel', devices),
                    'localID': ('channel', localIDs)
                },
                name= name,

                attrs= assign_attrs
            )
            return da


        ##===== Channel Lookup Array =======================================

        # 8) Construct Channel Lookup DataArray

    def construct_channel_lookup_da(self,
                                        use_typekey= None,
                                        channels_ref_dict=None,
                                        assign_attrs= None,
                                        name= None,
                                        channels=None
                                        ):
            '''
            Construct a Channel Lookup DataArray for a set of given channels.
            -----------------------------------

            Parameters:
                use_typekey (str or None): If provided, channel_ref_dict will be created using this type_key via create_channel_ref_dict, name will be set using get_channel_type_da_name, and channels will be set using get_channels_direct_from_ref_dict. If None, channels, channels_ref_dict, and name must be provided directly.
                channels (list): List of channel names to include in the DataArray. Obtained from self.channel_list.
                channels_ref_dict (dict): Dictionary mapping channel names to their reference information. Must be constructed using create_channel_ref_dict.
                assign_attrs (dict or None): Optional dictionary of attributes to assign to the DataArray. If None, default attributes will be used.
                name (str): Name of the DataArray. If None, defaults to 'Channel_lookup_matrix'. Should be provided using get_channel_type_da_name with an appropriate type_key matching channels_ref_dict.


            Returns:
                lookup_da (xr.DataArray): A DataArray representing the channel lookup matrix.
                lookup_devices (list): List of unique devices found in the channels_ref_dict.
                lookup_localIDs (list): List of unique local IDs found in the channels_ref_dict.
            '''
            #--------------------------------------------------------------------------------------------------------------------
            if use_typekey is not None:
                if use_typekey not in self.type_keys:
                    raise ValueError(f"use_typekey '{use_typekey}' is not valid.")
                
                if channels is not None:
                    raise ValueError("If use_typekey is provided, channels must not be provided directly.")

                if channels_ref_dict is not None:
                    raise ValueError("If use_typekey is provided, channels_ref_dict must not be provided directly.")
                
                if name is not None:
                    raise ValueError("If use_typekey is provided, name must not be provided directly.")
                
                channels_ref_dict= self.create_channel_ref_dict(type_key= use_typekey)
                name= self.get_channel_type_da_name(type_key= use_typekey)
                channels= self.channel_list

            if use_typekey is None:
                if channels is None:
                    raise ValueError("channels must be provided.")

                if channels_ref_dict is None:
                    raise ValueError("channels_ref_dict must be provided.")
                
            if name is None:
                name= 'Channel_lookup_matrix'

            if assign_attrs is None:
                assign_attrs = {
                    'description': 'Channel lookup matrix indicating presence of channels across devices and local IDs',
                    'source': 'Constructed using construct_channel_lookup_da function'
                }
            
            #--------------------------------------------------------------------------------------------------------------------

            lookup_devices= []
            lookup_localIDs= []

            for d, l in channels_ref_dict.values():
                if d not in lookup_devices:
                    lookup_devices.append(d)
                if l not in lookup_localIDs:
                    lookup_localIDs.append(l)

            lookup_arr= np.zeros((len(channels), len(lookup_devices), len(lookup_localIDs)), dtype= bool)

            for i, channel in enumerate(channels):
                look_device, look_localID = channels_ref_dict[channel]
                
                device_index= lookup_devices.index(look_device)
                localID_index= lookup_localIDs.index(look_localID)

                lookup_arr[i, device_index, localID_index] = True

            lookup_da= xr.DataArray(
                data= lookup_arr,
                dims= ['channel', 'device', 'localID'],
                coords= {
                    'channel': channels,
                    'device': lookup_devices,
                    'localID': lookup_localIDs
                },
                name= name,
                attrs= assign_attrs
            )
            return lookup_da, lookup_devices, lookup_localIDs



        ##===== Update Channel Type Data Array =============================
    
        # 13b v2) Update Channel Type DataArray

    def update_channel_type_da(self,
                                   use_typekey= None,
                                   channel_type_da= None, # Existing DataArray to update
                                   channel_ref_dict= None,     # {channel: (device, localID)}
                                   dict_of_devices_registers_dfs_dicts= None, # {device: {register_address: DataFrame(index= Time, columns= localIDs)}}
                                   device_register_dfs_dict= None, # {device: DataFrame(index= Time, columns= localIDs)}
                                   channel_type_registerIDs= None, # {type_key: {localID: register_address}}
                                   type_key= None,
                                   fill_value= None,
                                   verbose= False,

        ):
            '''
            Update channel_type_da at (Time, channel) positions using: dict_of_devices_registers_dfs_dicts[device][reg_address][localID].   

            Steps per channel:
                Map channel -> (device, localID) via channel_ref_dict
                Map localID -> reg_address via channel_type_registerIDs[type_key][localID]
                Pull source column; align Time to channel_type_da.Time (intersection)
                Optional fill within aligned rows; drop NaNs
                Collect columns into a (Time x channel) patch and assign once

            Notes:
                i. Optional. If provided, will overwrite value obtained when using use_typekey.
                ii. Optional. If None, will be obtained from self.dict_of_devices_registers_dfs_dicts, self.verbose, and self.channel_type_registerIDs, respectively.
                iii. Should always keep device_register_dfs_dict as None. This parameter is only retained for potential future flexibility/compatability requirements. If not None, then will cause issus with getting dict_of_devices_registers_dfs_dicts from self.dict_of_devices_registers_dfs_dicts and use_typekey functionality.
                iv. Should avoid type_key in place of use_typekey unless strictly necessary. If use_typekey and type_key are both provided, then use_typekey will take precedence and overwrite type_key. This is to ensure consistency across parameters. The functionality to NOT do use_typekey is only retained for greater flexibility.


            Parameters:
                use_typekey (str or None): If provided;
                    - channel_ref_dict will be created using this type_key via create_channel_ref_dict, unless channel_ref_dict is provided directly.
                    - type_key will be set to this value, Note: This will overwrite type_key if both are provided.
                    - channel_type_da will be set using construct_channel_type_da with this type_key, unless channel_type_da is provided directly.
                    - If None, channel_ref_dict, type_key, and channel_type_da must be provided directly.
                    It is recommended to provide use_typekey to ensure consistency across parameters. The functionality to NOT do use_typekey is only retained for greater flexibility.
                (i) channel_type_da (xr.DataArray): Existing DataArray to update. 
                (i) channel_ref_dict (dict): Dictionary mapping channel names to their reference information.
                (ii) dict_of_devices_registers_dfs_dicts (dict): Nested dictionary of devices and their register DataFrames.
                (iii) device_register_dfs_dict (dict or None): Optional pre-constructed dict of device DataFrames to use instead of nested dict.
                (ii) channel_type_registerIDs (dict): Mapping of type_key to localID to register_address.
                (iv) type_key (str): The type_key corresponding to the channel_type_da.
                fill_value: Value to use for missing data. If None, will use NaN for floats and False for bools. Recommended to use self.fill_value for consistency.
                (ii) verbose (bool or None): If True, prints detailed output during the update process. If None, defaults to self.verbose.

            Returns:
                xr.DataArray: Updated DataArray with new data.
            '''
            
            #--------------------------------------------------------------------------------------------------------------------
            if verbose is None:
                verbose= self.verbose
            
            
            
            if use_typekey is not None:
                if use_typekey not in self.type_keys:
                    raise ValueError(f"use_typekey '{use_typekey}' is not valid.")
                
                if channel_ref_dict is None:
                    channel_ref_dict= self.create_channel_ref_dict(type_key= use_typekey)
                elif channel_ref_dict is not None and verbose:
                    # print("Warning: channel_ref_dict provided directly will override that obtained from create_channel_ref_dict.")
                    # See if provided channel_ref_dict match those which would be obtained from create_channel_ref_dict
                    if set(channel_ref_dict.keys()) != set(self.create_channel_ref_dict(type_key=use_typekey).keys()):
                        print("Warning: Provided channel_ref_dict do not match those from create_channel_ref_dict.")
                
                if type_key is None:
                    type_key= use_typekey
                elif type_key is not None and verbose:
                    # print("Warning: type_key provided directly will override that obtained from use_typekey.")
                    if type_key != use_typekey:
                        print("Warning: Provided type_key does not match use_typekey, overriding with use_typekey.")
                        type_key= use_typekey
                    
                if channel_type_da is None:
                    channel_type_da= self.construct_channel_type_da(use_typekey= use_typekey)
                elif channel_type_da is not None and verbose:
                    # print("Warning: channel_type_da provided directly will override that obtained from construct_channel_type_da.")
                    # See if provided channel_type_da matches that which would be obtained from construct_channel_type_da
                    constructed_da= self.construct_channel_type_da(use_typekey= use_typekey)
                    if not channel_type_da.equals(constructed_da):
                        print("Warning: Provided channel_type_da does not match that from construct_channel_type_da.")
                
            if dict_of_devices_registers_dfs_dicts is None and device_register_dfs_dict is None:
                dict_of_devices_registers_dfs_dicts= self.dict_of_devices_registers_dfs_dicts
                if verbose:
                    print("Using self.dict_of_devices_registers_dfs_dicts for data source.")
            
            if channel_type_registerIDs is None:
                channel_type_registerIDs= self.channel_type_registerIDs
                if verbose:
                    print("Using self.channel_type_registerIDs for register mapping.")
            elif channel_type_registerIDs is not None and verbose:
                # print("Warning: channel_type_registerIDs provided directly will override self.channel_type_registerIDs.")
                if channel_type_registerIDs != self.channel_type_registerIDs:
                    print("Warning: Provided channel_type_registerIDs do not match self.channel_type_registerIDs.")
            
            #--------------------------------------------------------------------------------------------------------------------


            #===== Error Checks =====
            if dict_of_devices_registers_dfs_dicts is None:
                raise ValueError("dict_of_devices_registers_dfs_dicts is required for this update path.")
            if channel_type_registerIDs is None or type_key is None:
                raise ValueError("channel_type_registerIDs and type_key are required.")
            if type_key not in channel_type_registerIDs:
                raise ValueError(f"type_key '{type_key}' not found in channel_type_registerIDs.")
            if channel_type_da is None:
                raise ValueError("channel_type_da is required.")
            if channel_ref_dict is None:
                raise ValueError("channel_ref_dict is required.")
            if not isinstance(channel_type_da, xr.DataArray):
                raise ValueError("channel_type_da must be an xarray DataArray.")
            

            #===== Setup =====

            ctda= channel_type_da
            ctda_time= ctda.get_index('Time')
            ctda_channels= ctda.coords['channel'].to_index()

            patch_cols= {} # channel -> aligned pandas column

            #===== Main Loop Over Channels =====

            for channel, (device, localID) in channel_ref_dict.items():

                reg_addr= channel_type_registerIDs[type_key][localID]                                                   # get reg addr for localID for given type_key

                if device not in list(dict_of_devices_registers_dfs_dicts.keys()):
                    if verbose:
                        print(f'Channel {channel} on device {device} localID {localID} has no device data; skipping.')
                    continue

                if reg_addr in list(dict_of_devices_registers_dfs_dicts[device].keys()):
                    source_col= dict_of_devices_registers_dfs_dicts[device][reg_addr][localID]                          # Get source column (pd Series-like) indexed by Time
                else:
                    if verbose:
                        print(f'Channel {channel} on device {device} localID {localID} has no register address {reg_addr}; skipping.')
                    continue

                # source_col= dict_of_devices_registers_dfs_dicts[device][reg_addr][localID]  # Get source column (pd Series-like) indexed by Time

                time_index= ctda_time.intersection(source_col.index)                                                    # Aligns Time by intersection with ctda.Time
                if len(time_index) == 0:                                                                                # Skip if no overlap in Time
                    if verbose:
                        print(f'Channel {channel} on device {device} localID {localID} has no overlapping Time with channel_type_da; skipping.')
                    continue

                vals= source_col.reindex(time_index)                                                                    # Re-index to aligned Time, keeps only rows with timestamps present in time_index and with same order
                if fill_value is not None:
                    vals= vals.fillna(fill_value)                                                                       # Optional fill within aligned rows
                
                vals= vals.dropna()                                                                                     # Drop NaNs to avoid overwriting with NaN
                if vals.empty:                                                                                          # Skip if no values remain after dropna
                    if verbose:
                        print(f'Channel {channel} on device {device} localID {localID} has no values after alignment; skipping.')
                    continue

                if channel not in ctda_channels:                                                                        # Skip if channel not in ctda_channels
                    if verbose:
                        print(f'Channel {channel} not found in channel_type_da; skipping.')
                    continue

                patch_cols[channel]= vals.rename(channel)                                                               # Store aligned column for this channel

            if not patch_cols:
                if verbose:
                    print("No valid data found to update channel_type_da; returning original.")
                return ctda
            
            #===== Construct Patch DataFrame =====
            patch_df= pd.concat(patch_cols.values(), axis=1)                                                            # Combine all columns into a DataFrame indexed by Time
            patch_df= patch_df.reindex(index= ctda_time, 
                                    columns= [c for c in ctda_channels if c in patch_df.columns])                    # Reindex to ctda_time and ctda_channels, keeps only channels present in patch_df
            
            if verbose:
                print(f'Constructed patch DataFrame with shape: {patch_df.shape} for updating channel_type_da.')
                display(patch_df)
            #===== Update DataArray =====

            #===== Update DataArray (only write where values exist) =====
            for ch in patch_df.columns:
                col = patch_df[ch].dropna()
                if not col.empty:
                    ctda.loc[dict(Time=col.index, channel=ch)] = col.astype(ctda.dtype).values

            if verbose:
                print(f'Updated channel_type_da with data from {len(patch_cols)} channels.')
                print(f'New shape: {ctda.shape}')

            return ctda
        

        ##===== Create and Update All Channel Type DataArrays ==============

    def get_channel_type_outputs(self,
                                        update= True,
                                        use_typekey= None,
                                        fill_value= None,
                                        verbose= None,
                                        ctda_assign_attrs= None,
                                        lookup_assign_attrs= None
                                         ):
            '''
            Get all channel type outputs for the specified parameters.
            -----------------------------------

            Parameters:
                update (bool): If True, updates the channel_type_da using update_channel_type_da. If False, only constructs the channel_type_da without updating.
                use_typekey (str or None): Type key to use for constructing and updating the channel_type_da. Must be one of self.type_keys.
                fill_value: Value to use for missing data when updating the channel_type_da. If None, will use NaN for floats and False for bools. Recommended to use self.fill_value for consistency.
                verbose (bool or None): If True, prints detailed output during processing. If None, defaults to self.verbose.
                ctda_assign_attrs (dict or None): Optional dictionary of attributes to assign to the channel_type_da. If None, default attributes will be used.
                lookup_assign_attrs (dict or None): Optional dictionary of attributes to assign to the channel_lookup_da. If None, default attributes will be used.
            Returns:
                channel_type_da (xr.DataArray): The constructed (and possibly updated) channel type DataArray.
                channel_lookup_da (xr.DataArray): The constructed channel lookup DataArray.
                channel_ref_dict (dict): The channel reference dictionary used for construction and updating.
            '''

            if verbose is None:
                verbose= self.verbose
            if fill_value is None:
                fill_value= self.fill_value

            if use_typekey is None:
                raise ValueError("use_typekey must be provided.")
            if use_typekey not in self.type_keys:
                raise ValueError(f"use_typekey '{use_typekey}' is not valid.")
            if ctda_assign_attrs is None:
                ctda_assign_attrs= {
                    'description': f'DataArray for channels of type {use_typekey}',
                    'source': 'Constructed using get_all_channel_type_outputs function'
                }
            if lookup_assign_attrs is None:
                lookup_assign_attrs= {
                    'description': 'Channel lookup matrix indicating presence of channels across devices and local IDs',
                    'source': 'Constructed using get_all_channel_type_outputs function'
                }
            
            channel_ref_dict= self.create_channel_ref_dict(type_key= use_typekey)

            if verbose:
                print(f'Constructed channel_ref_dict with {len(channel_ref_dict)} entries.')

            # Create and update channel_type_da

            channel_type_da= self.construct_channel_type_da(values= None,
                                                            use_typekey= use_typekey,
                                                            channels= None,
                                                            channels_ref_dict= None,
                                                            times= None,
                                                            assign_attrs= ctda_assign_attrs,
                                                            name= None,
                                                            verbose= verbose
                )


            if update:
                channel_type_da= self.update_channel_type_da(use_typekey= use_typekey,
                                                            channel_type_da= channel_type_da,
                                                            channel_ref_dict= None,
                                                            dict_of_devices_registers_dfs_dicts= None,
                                                            device_register_dfs_dict= None,
                                                            channel_type_registerIDs= None,
                                                            type_key= None,
                                                            fill_value= fill_value,
                                                            verbose= verbose
                                                             )


            # Create channel_lookup_da
            channel_lookup_da= self.construct_channel_lookup_da(use_typekey= use_typekey,
                                                            # channels_ref_dict= channel_ref_dict,
                                                            assign_attrs= lookup_assign_attrs,
                                                            # name= None,

                                                                )



            return channel_type_da, channel_lookup_da, channel_ref_dict

# ################################################################################
# # Imports
# ################################################################################


# ################################################################################





# ################################################################################
# # Imports
# ################################################################################


# ################################################################################





# ################################################################################
# # Imports
# ################################################################################


# ################################################################################





# ################################################################################################################################################################
# # Wrappers for renamed, moved, or deprecated functions
# ################################################################################################################################################################




# ################################################################################
# # Imports
# ################################################################################


# ################################################################################





# ################################################################################
# # Imports
# ################################################################################


# ################################################################################





# ################################################################################
# # Imports
# ################################################################################


# ################################################################################





# ################################################################################
# # Imports
# ################################################################################


# ################################################################################





# ################################################################################
# # Imports
# ################################################################################


# ################################################################################

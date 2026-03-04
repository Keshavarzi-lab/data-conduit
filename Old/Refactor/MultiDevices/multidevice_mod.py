'''
Uitilities for handling multi-device data arrays.


'''



########################################################################################################################
# Imports
########################################################################################################################

import pandas as pd
import numpy as np
import xarray as xr
from collections.abc import Mapping
from pathlib import Path


########################################################################################################################









########################################################################################################################
# Private Helper Functions
########################################################################################################################




#===============================================================================
# 1| 
#===============================================================================



#===============================================================================




#===============================================================================
# 2| 
#===============================================================================



#===============================================================================




#===============================================================================
# 3| 
#===============================================================================



#===============================================================================




#===============================================================================
# 4| 
#===============================================================================



#===============================================================================





########################################################################################################################
"""
class BehaviorNosepoke:
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

"""




########################################################################################################################
# Public Functions
########################################################################################################################




################################################################################
# Base MultiDevice Class
################################################################################

class MultiDevice:
    '''
    Class to manage loading and processing of multi-device data from HARP devices.
    -----------------------------------

    '''

    def __init__(self,
                # Virtual Map-related Parameters
                 virtual_maps: dict = None,
                 global_coord_name: str = 'global_coords',
                 virtual_coord_names: list = None,
                 dict_of: str = 'dicts',
                 data_keys: list | None = None,                             # List of data keys to load (list of str). E.g. ['Behavior', 'Wheel',...]
                 data_registerIDs: dict[str, dict[str, str]] | None = None, # Dict of data keys to dict of localID to register address (dict of str: dict of str: str). E.g. {'Activations': {'DIPort0': '32', 'DIPort1': '32',...},...}
                 # DataFrame-related Parameters
                 experiment_directory_path: str | None = None,              # Path to experimental directory (str or Path)
                 harp_device_yaml_path: str | None = None,                  # Path to harp_device yaml file (str or Path)
                 device_type: str = 'Behavior',                             # Device type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc.
                 device_list: list | None = None,                           # List of device names to load (list of str)
                 device_IDs: dict | None = None,                            # Dict of device names to IDs (dict of str: str)
                 device_registers_dict: dict | None = None,                 # Dict of device names to list of register addresses (dict of str: list of str). dict= {f'{device_list[n]}': [registers],...}
                 lookfor: str | None = None,                                # If provided, will select register paths containing this substring if multiple possible matches exist.
                 validate: bool = False,                                    # If True, runs validation checks on devices, device IDs, device registers, and register channels. If False, skips validation.
                 # DataArray-related Parameters
                 data_array_names: list | None = None,                     # Names to use for DataArrays (list). If None, will use f'{selected_type_key}_data'.
                 data_array_attrs: dict | None = None,                      # Dict of attributes to add to DataArrays (dict of list: dict). If None, will use default attributes. 
                 lookup_array_names: list | None = None,                   # Names to use for lookup arrays (list). If None, will use f'{selected_type_key}_lookup_matrix'.
                 lookup_array_attrs: dict | None = None,                    # Dict of attributes to add to lookup arrays (dict of list: dict). If None, will use default attributes. 
                 test_values: bool = False,                                 # If True, use test helper values instead of NaNs for channel type DataArrays (bool)
                 fill_value = None,                                         # Value to use for missing data when updating channel type DataArrays (e.g., 0, False). If None, will use NaN for floats and False for bools.
                 verbose: bool = False,                                     # If True, prints detailed output during processing (bool)
                 ):

        super(MultiDevice, self).__init__()


        ##===== Virtual Map Parameters =====================================
        self.virtual_maps = virtual_maps
        self.global_coord_name = global_coord_name
        self.virtual_coord_names = virtual_coord_names
        self.dict_of = dict_of
        self.data_keys = data_keys

        
        ##===== DataFrame Parameters =======================================
        self.experiment_directory_path = experiment_directory_path
        self.harp_device_yaml_path = harp_device_yaml_path
        self.device_type = device_type
        self.device_list = device_list
        self.device_IDs = device_IDs
        self.device_registers_dict = device_registers_dict
        self.lookfor = lookfor
        self.validate = validate

        ##===== DataArray Parameters =======================================
        self.data_array_names = data_array_names
        self.data_array_attrs = data_array_attrs
        self.lookup_array_names = lookup_array_names
        self.lookup_array_attrs = lookup_array_attrs
        self.test_values = test_values
        self.fill_value = fill_value
        self.verbose = verbose



        ##===== Setup ======================================================
        if virtual_maps is not None and data_keys is not None:
            self.peripheral_registerIDs = {
                dk: self.infer_peripheral_registerIDs(data_key=dk, virtual_map=virtual_maps)[dk]
                for dk in data_keys
            }
        elif data_registerIDs is not None and data_keys is not None:
            self.peripheral_registerIDs = {
                dk: self.infer_peripheral_registerIDs(data_key=dk, data_registerIDs=data_registerIDs)[dk]
                for dk in data_keys
            }
        else:
            self.peripheral_registerIDs = None
            if verbose: print(f'''Warning: No virtual_map or data_registerIDs provided for inferring peripheral register IDs. 
                                peripheral_registerIDs will be set to None, which may cause issues for timestamp collection if 
                              using collect_timestamps_nested_dict.''')
                

        ##===== Construct Outputs ==========================================
        
        if experiment_directory_path is not None and data_keys is not None:
            results = self.construct_all_data_outputs(verbose=verbose)

            self.data_arrays = {}
            self.lookup_arrays = {}
            self.lookup_virtual_coords = {}

            for dk, (da, lookup_da, lookup_vc) in results.items():
                self.data_arrays[dk] = da
                self.lookup_arrays[dk] = lookup_da
                self.lookup_virtual_coords[dk] = lookup_vc

        else:
            self.data_arrays = None
            self.lookup_arrays = None
            self.lookup_virtual_coords = None
            print(f"Skipping auto-construction: experiment_directory_path or data_keys not provided.")
    
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
    # 3| Collect Registers
    #===========================================================================
    def _collect_registers(self,
                           harp_device_yaml_path: str | None = None,
                           register_addresses: list | None = None,
                           verbose: bool | None = None,
    ):
        '''
        Wrapper for collect_registers function from harp_extender_core. 
        By default, uses self.harp_device_yaml_path, self.verbose. 
        Requires register_addresses to be passed in, which can be obtained from self.device_registers_dict or passed in directly.
        ------------------
        
        Parameters:
            harp_device_yaml_path: yaml file path
            register_addresses: single int/string OR list of ints/strings
            verbose: If True, prints additional information.
        
        Returns: 
            Single string (if input was scalar) OR List of strings (if input was list)
        '''
        from Refactor.HarpExtender.harp_extender_core import collect_registers
        if harp_device_yaml_path is not None:
            harp_device_yaml_path = self.harp_device_yaml_path
        if verbose is None:
            verbose = self.verbose
        
        if register_addresses is None:
            raise ValueError('register_addresses must be provided to collect_registers function. This can be obtained from self.device_registers_dict or passed in directly.')
        
        return collect_registers(harp_device_yaml_path, register_addresses, verbose)

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



    #===========================================================================
    # 4| Infer Peripheral Register IDs
    #===========================================================================
    def infer_peripheral_registerIDs(self,
                                    data_key: str,
                                    virtual_map: dict | None = None,
                                    data_registerIDs: dict[str, dict[str, str]] | None = None,
                                    ) -> dict[str, dict[str, str]]:
        """
        Extract peripheral register IDs for a given data key, in the format
        expected by collect_timestamps_nested_dict.

        Supports two input modes:
            1. virtual_map: extracts {localID: register} from virtual_map[data_key]
            2. data_registerIDs: returns {data_key: data_registerIDs[data_key]} directly

        Parameters
        ----------
        data_key : str
            The data key to extract peripheral register IDs for (e.g., 'Activations', 'LED').

        virtual_map : dict or None
            Multi-map virtual map structure:
            {data_key: {unified_name: {'device': str, 'register': str, 'localID': str}, ...}, ...}

        data_registerIDs : dict or None
            Legacy register ID mapping:
            {data_key: {localID: register_address}}

        Returns
        -------
        dict[str, dict[str, str]]
            {data_key: {localID: register_address}}.
            Compatible with the ``peripheral_registerIDs`` parameter of
            ``collect_timestamps_nested_dict``.

        Raises
        ------
        ValueError
            If both or neither of virtual_map and data_registerIDs are provided.
        KeyError
            If data_key is not found in the provided mapping.

        Examples
        --------
        >>> # From virtual_map
        >>> get_peripheral_registerIDs('Activations', virtual_map=vmap)
        {'Activations': {'DIPort0': '32', 'DIPort1': '32'}}

        >>> # From data_registerIDs (legacy)
        >>> get_peripheral_registerIDs('Activations', data_registerIDs=reg_ids)
        {'Activations': {'DIPort0': '32', 'DIPort1': '32'}}
        """
        if virtual_map is not None and data_registerIDs is not None:
            raise ValueError("Specify either 'virtual_map' or 'data_registerIDs', not both.")
        if virtual_map is None and data_registerIDs is None:
            raise ValueError("Must specify either 'virtual_map' or 'data_registerIDs'.")

        if virtual_map is not None:
            if data_key not in virtual_map:
                raise KeyError(f"data_key '{data_key}' not in virtual_map. Available: {list(virtual_map.keys())}")
            return {
                data_key: {
                    coords['localID']: coords['register']
                    for coords in virtual_map[data_key].values()
                    if 'localID' in coords and 'register' in coords
                }
            }

        else:
            if data_key not in data_registerIDs:
                raise KeyError(f"data_key '{data_key}' not in data_registerIDs. Available: {list(data_registerIDs.keys())}")
            return {data_key: data_registerIDs[data_key]}
    
    #===========================================================================
    # 5| Collect Timestamps
    #===========================================================================
    def _collect_timestamps(self,
                            df: pd.DataFrame | xr.DataArray | None = None,
                            timestamp_name: str | None = None,
                            time_location: str | None = None,
                            verbose: bool | None = None,
                            ) -> pd.Series | None:
        '''
        Wrapper for collect_timestamps function from timestamps module. Collects and processes timestamps from a given DataFrame or DataArray.
        -----------------------

        Parameters:
        -----------
        df : pd.DataFrame | xr.DataArray | None
            The DataFrame or DataArray containing the timestamps.
        timestamp_name : str | None
            The name of the column containing the timestamps. If None, defaults to 'Time'. Note: case-insensitive.
        time_location : str | None
            The location of the timestamps, either 'columns' or 'index'. If None, will attempt index first, then columns.
        verbose : bool
            If True, prints warnings and information during processing.

        Returns:
        --------
        pd.Series | None
            A Series of processed timestamps, or None if the DataFrame is None.
        '''

        from Refactor.Timestamps.timestamps import collect_timestamps
        if df is None:
            if verbose:
                print('DataFrame is None, cannot collect timestamps.')
            return None
        if verbose is None:
            verbose = self.verbose
        
        
        return collect_timestamps(df = df, timestamp_name = timestamp_name, time_location = time_location, verbose = verbose )

    def _collect_timestamps_dict(self,
                                 dfs: dict[str, pd.DataFrame | xr.DataArray],
                                 timestamp_name: str | None = None,
                                 time_location: str | None = None,
                                 verbose: bool | None = None,
                                 return_type: str = 'list',  # 'dict' or 'list'
                                 ) -> dict[str, pd.Series] | list[pd.Series]:
        '''
        Wrapper for collect_timestamps_dict function from timestamps module. Collects and processes timestamps from a dictionary of DataFrames or DataArrays.
        -----------------------
        Parameters:
            dfs (dict): A dictionary where keys are identifiers and values are pandas DataFrames.
            timestamp_name (str, optional): The name of the column containing the timestamps. If None, defaults to 'Time'. Note: case-insensitive.
            verbose (bool, optional): If True, prints warnings and information during processing. Default is True.
            return_type (str, optional): If 'list', returns a sorted list of all timestamps. If 'dict', returns a dictionary of timestamps per DataFrame. Default is 'list'. If both are needed, call the function twice.

        Returns:
            list of all timestamps from each DataFrame in the dictionary, sorted in ascending order.
            dict where keys are the same as df_dict and values are the timestamps from each DataFrame.
        '''
        from Refactor.Timestamps.timestamps import collect_timestamps_dict
        
        if verbose is None:     verbose = self.verbose
        
        return collect_timestamps_dict(dfs = dfs,
                                       timestamp_name = timestamp_name,
                                       time_location = time_location,
                                       verbose = verbose,
                                       return_type = return_type
                                       )
    
    def _collect_timestamps_nested_dict(self,
                                        dfs_dict: dict[str, dict[str, pd.DataFrame | xr.DataArray]],
                                        data_key: str | None = None,
                                        device_list: list[str] | None = None,
                                        peripheral_registerIDs: dict[str, dict[str, str]] | None = None,
                                        return_type: str = 'list', # 'dict' or 'list' or 'both'
                                        verbose: bool | None = None,
                                        ) -> dict[str, pd.Series] | list[pd.Series] | tuple[dict[str, pd.Series], list[pd.Series]]:
        '''
        Wrapper for collect_timestamps_nested_dict function from timestamps module. Collects and processes timestamps from a nested dictionary of DataFrames or DataArrays.
        -----------------------
        
        Parameters:
            dfs_dict (dict): dfs_dict {device: {register_address: DataFrame(Time index, localID columns)}}
            data_key (str): optional peripheral type to use (e.g., the global type_key) if using peripheral_registerIDs
            devices (list or None): optional subset of devices to include; defaults to all in dfs_dict
            peripheral_registerIDs (dict or None): optional subset of peripheral register IDs to include; defaults to all for each device
            return_type (str): 'list', 'dict', or 'both' (passed through to collect_timestamps_dict)
            verbose (bool): If True, prints warnings and information during processing.
        
        Returns:
            list | dict | (list, dict): timestamps per return_type

        '''

        from Refactor.Timestamps.timestamps import collect_timestamps_nested_dict

        if verbose is None:     verbose = self.verbose
        if dfs_dict is None: 
                dfs_dict = self.dfs_dict
        if data_key is None:
                if verbose: print(f'Warning: No data_key provided for collect_timestamps_nested_dict')
        if verbose is None:
                verbose = self.verbose
        
        if peripheral_registerIDs is None:
            if self.peripheral_registerIDs is not None:
                peripheral_registerIDs = self.peripheral_registerIDs
            else:
                if verbose: print(f'''Warning: No peripheral_registerIDs provided for collect_timestamps_nested_dict. Will attempt to infer from virtual_map if available, 
                                    but may cause issues if virtual_map is not properly structured or does not contain the necessary information.''')
                if self.virtual_maps is not None and data_key is not None:
                    peripheral_registerIDs = self.infer_peripheral_registerIDs(data_key=data_key, virtual_map=self.virtual_maps)
                else:
                    if verbose: print(f'''Warning: No virtual_maps or data_key available to infer peripheral_registerIDs. peripheral_registerIDs will be set to None, which 
                                      may cause issues for timestamp collection if using collect_timestamps_nested_dict.''')
                    peripheral_registerIDs = None
        
        return collect_timestamps_nested_dict(dfs_dict = dfs_dict,
                                              data_key = data_key,
                                              devices = device_list,
                                              peripheral_registerIDs = peripheral_registerIDs,
                                              return_type = return_type,
                                              verbose = verbose
                                                )      
                                        
    #===========================================================================



        
    #===========================================================================
    # 6| Construct Base DataArray
    #===========================================================================
    def _construct_data_array(self,
                              virtual_map: dict= None,
                              global_coord_name: str | None= None,
                              virtual_coord_names: list | None= None,
                              times: list | np.ndarray= None,
                              dict_of: str | None= None,
                              values: list | np.ndarray | None= None,
                              name: str | None = None,
                              test_values: bool | None = None,
                              verbose: bool | None = None,
                              da_attrs: dict | None= None
                              ) -> xr.DataArray:
        '''
        Wrapper for construct_data_array function from lookup_arrays module. Constructs a DataArray for a given type key based on the provided parameters.
        -----------------------
            
        virtual_map formats:
            dict_of="dicts":  {global: {vcoord_name: vcoord_value, ...}, ...}
            dict_of="tuples": {global: (v0, v1, ...), ...} aligned with virtual_coord_names
        -------------------------------

        Parameters:
            virtual_map: dict
                A dictionary mapping global coordinates to virtual coordinates.
            global_coord_name: str
                The name of the global coordinate dimension.
            virtual_coord_names: list
                A list of names for the virtual coordinate dimensions.
            times: list | np.ndarray
                A list or array of timepoints to include in the DataArray. Should be obtained using collect_timestamps_dict() or collect_timestamps_nested_dict() with  return_type= 'list'.
            dict_of: str
                Specifies the format of virtual_map_dict. Options are 'dicts' or 'tuples'.
            values: list | np.ndarray | None
                Values to populate the DataArray. If None, a placeholder array of NaNs will be created.
            name: str
                The name of the DataArray.
            test_values: bool
                If True, function will use test values instead of the provided values. Default is False. Useful for debugging.
            verbose: bool
                If True, function will print additional information during execution. Default is False.
            da_attrs: dict | None
                A dictionary of attributes to assign to the DataArray. Default is None.

        Returns:
            xr.DataArray
                A DataArray with dimensions for time, global coordinates, and virtual coordinates.
        '''
        # ToDo: Need to replace Lookup_Arrays with Lookup_Arrays_mod as former is deprecated
        from Refactor.LookupArrays.Lookup_Arrays_mod import construct_data_array
        

        if virtual_map is None:
            raise ValueError(f'''Virtual map must be specifically provided to construct_data_array function as it is not stored as an attribute\n
                             in the MultiDevice class. This is because the virtual map may differ for each data key, so it cannot be set as a \n
                             single attribute for the class. Please provide the appropriate virtual map for the data key you are constructing \n
                             the DataArray for.''')
        if name is None:
            if verbose: print('No name provided for DataArray, using default name "data_array".')
            name = 'data_array'
        
        if times is None:
            raise ValueError(f'''Times must be provided to construct_data_array function as they are not stored as an attribute\n
                             in the MultiDevice class. This is because the times may differ for each data key, so they cannot be set as a \n
                             single attribute for the class. Please provide the appropriate times for the data key you are constructing \n
                             the DataArray for.''')        
        

        if global_coord_name is None:           global_coord_name = self.global_coord_name
        if virtual_coord_names is None:         virtual_coord_names = self.virtual_coord_names
        if dict_of is None:                     dict_of = self.dict_of
        if test_values is None:                 test_values = self.test_values
        if verbose is None:                     verbose = self.verbose
        if da_attrs is None:                    da_attrs = self.data_array_attrs

        return construct_data_array(virtual_map = virtual_map,
                                    global_coord_name = global_coord_name,
                                    virtual_coord_names = virtual_coord_names,
                                    dict_of = dict_of,
                                    times = times,
                                    values = values,
                                    name = name,
                                    test_values = test_values,
                                    verbose = verbose,
                                    da_attrs = da_attrs
                                    )
    #===========================================================================





    #===========================================================================
    # 7| Construct Lookup Array
    #===========================================================================
    def _construct_lookup_array(self,
                               virtual_map: dict= None,
                               global_coord_name: str | None = None,
                               virtual_coord_names: list | None = None,
                               dict_of: str | None = None,
                               name: str | None = None,
                               verbose: bool | None = None,
                               lookup_attrs: dict | None= None
                               ) -> tuple[xr.DataArray, dict[str, list]]:
        '''
        Wrapper for construct_lookup_array function from lookup_arrays module. Constructs a lookup array DataArray and corresponding lookup dictionary for a given type key based on the provided parameters.
        -----------------------
            
        virtual_map formats:
            dict_of="dicts":  {global: {vcoord_name: vcoord_value, ...}, ...}
            dict_of="tuples": {global: (v0, v1, ...), ...} aligned with virtual_coord_names
        -------------------------------
        Parameters:
            virtual_map: dict
                A dictionary mapping global coordinates to virtual coordinates.
            global_coord_name: str
                The name of the global coordinate dimension.
            virtual_coord_names: list
                A list of names for the virtual coordinate dimensions.
            dict_of: str
                Specifies the format of virtual_map_dict. Options are 'dicts' or 'tuples'.
            name: str
                The name of the DataArray.
            verbose: bool
                If True, function will print additional information during execution. Default is False.
            lookup_attrs: dict | None
                A dictionary of attributes to assign to the DataArray. Default is None.

        Returns:
            lookup_da (xr.DataArray): 
                A DataArray representing the lookup matrix.
            lookup_virtual_coords (dict):
                A dictionary containing lists of unique virtual coordinate values for each virtual coordinate dimension. Keys are virtual coordinate names, values are lists of unique values. 
        '''
        # ToDo: Need to replace Lookup_Arrays with Lookup_Arrays_mod as former is deprecated
        from Refactor.LookupArrays.Lookup_Arrays_mod import construct_lookup_array
        
        if virtual_map is None:
            raise ValueError(f'''Virtual map must be specifically provided to construct_lookup_array function as it is not stored as an attribute\n
                             in the MultiDevice class. This is because the virtual map may differ for each data key, so it cannot be set as a \n
                             single attribute for the class. Please provide the appropriate virtual map for the data key you are constructing \n
                             the lookup array for.''')
        if name is None:
            if verbose: print('No name provided for lookup DataArray, using default name "lookup_array".')
            name = 'lookup_array'
        
        if global_coord_name is None:           global_coord_name = self.global_coord_name
        if virtual_coord_names is None:         virtual_coord_names = self.virtual_coord_names
        if dict_of is None:                     dict_of = self.dict_of
        if verbose is None:                     verbose = self.verbose
        if lookup_attrs is None:                lookup_attrs = self.lookup_array_attrs

        return construct_lookup_array(virtual_map = virtual_map,
                                      global_coord_name = global_coord_name,
                                      virtual_coord_names = virtual_coord_names,
                                      dict_of = dict_of,
                                      name = name,
                                      verbose = verbose,
                                      lookup_attrs = lookup_attrs
                                    )
    #===========================================================================


        
    #===========================================================================
    # 8| Update DataArray
    #===========================================================================
    def _update_data_array(self,
                           data_array: xr.DataArray = None,
                           virtual_map: dict = None,
                           dict_of: str | None = None,
                           dfs_dict: dict = None,
                           fill_value: any = None,
                           verbose: bool | None = None
                           ) -> xr.DataArray:
        '''
        Wrapper for update_data_array function from lookup_arrays module. Format of dfs_dict and virtual_map should match, i.e. if dfs_dict is indexable as [global]: [virtual_1][virtual_2].
        -----------------------
           
        Parameters:
            data_array: xr.DataArray
                The DataArray to be updated. Should have dimensions for time and global coordinates, and virtual coordinate information in its coordinates.
            virtual_map: dict
                A dictionary mapping global coordinates to virtual coordinates. Format should match the structure of dfs_dict for lookups. Can be in dict-of-dicts or dict-of-tuples format.
            dict_of: str
                Specifies the format of virtual_map and dfs_dict. Options are 'dicts' or 'tuples'. Default is 'dicts'.
            dfs_dict: dict
                A nested dictionary containing the source data for updating the DataArray. Should be structured to allow lookups based on the virtual coordinates specified in virtual_map. For example, if virtual_map is dict-of-dicts with virtual coordinates 'device' and 'localID', dfs_dict should be structured as dfs_dict[device][localID][global_coord] to retrieve the value for a given global coordinate.
            fill_value: any
                A value to use for filling entries in the DataArray when lookups fail or when virtual coordinate information is missing. Default is np.nan.
            verbose: bool
                If True, function will print additional information during execution. Default is False.
        
        Returns:
            xr.DataArray
                The updated DataArray with values filled in from dfs_dict based on the mapping in virtual_map
        '''
        # ToDo: Need to replace Lookup_Arrays with Lookup_Arrays_mod as former is deprecated
        from Refactor.LookupArrays.Lookup_Arrays_mod import update_data_array
        
        if data_array is None:      raise ValueError('data_array must be provided to update_data_array function.')
        if virtual_map is None:     raise ValueError('virtual_map must be provided to update_data_array function.')
        if dfs_dict is None:        raise ValueError('dfs_dict must be provided to update_data_array function.')

        if dict_of is None:         dict_of = self.dict_of
        if fill_value is None:      fill_value = self.fill_value
        if verbose is None:         verbose = self.verbose

        return update_data_array(data_array = data_array,
                                 virtual_map = virtual_map,
                                 dict_of = dict_of,
                                 dfs_dict = dfs_dict,
                                 fill_value = fill_value,
                                 verbose = verbose
                                 )

    #===========================================================================


        
    #===========================================================================
    # 9| Master Function to Load and Process Data for a Given Type Key
    #===========================================================================
    def construct_data_outputs(self,
                               # --- Data key / virtual map ---
                               data_key: str | None = None,
                               virtual_map: dict | None = None,

                               # --- _collect_device_dfs args ---
                               dfs_dict: dict | None = None,
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
                               validate: bool = False,

                               # --- _collect_timestamps_nested_dict args ---
                               peripheral_registerIDs: dict[str, dict[str, str]] | None = None,
                               return_type: str = 'list',

                               # --- _construct_data_array args ---
                               data_array_name: str | None = None,
                               da_attrs: dict | None = None,

                               # --- _construct_lookup_array args ---
                               lookup_array_name: str | None = None,
                               lookup_attrs: dict | None = None,

                               # --- _update_data_array args ---
                               fill_value: any  = None,

                               # --- General ---
                               verbose: bool | None = None,
                               ) -> tuple[xr.DataArray, xr.DataArray, dict]:
        '''
        Master function to construct data_array, lookup_array, and update 
        data_array for a single data_key (or no data_key).

        Steps:
            1. Resolve virtual_map for this data_key.
            2. Collect nested dictionary of DataFrames using _collect_device_dfs() (if not provided).
            3. Collect timestamps using _collect_timestamps_nested_dict().
            4. Construct base DataArray using _construct_data_array().
            5. Construct lookup array using _construct_lookup_array().
            6. Update DataArray with values from DataFrames using _update_data_array().
        
        Parameters
        ----------
        data_key : str or None
            The data key to process (e.g., 'Activations', 'LED'). 
            If None, assumes single-map mode.
        virtual_map : dict or None
            The virtual map for this specific data_key. 
            {unified_name: {vcoord: value, ...}, ...}
            If None, will attempt to extract from self.virtual_maps using data_key.

        dfs_dict : dict or None
            Pre-loaded nested dict {device: {register: DataFrame}}.
            If None, will call _collect_device_dfs() with the following args.
        experiment_directory_path : str or Path or None
            Path to experiment directory. Falls back to self attribute.
        harp_device_yaml_path : str or Path or None
            Path to harp device yaml. Falls back to self attribute.
        device_type : str or None
            Device type string. Falls back to self attribute.
        device_list : list or None
            List of device names. Falls back to self attribute.
        device_IDs : dict or None
            Device ID mapping. Falls back to self attribute.
        device_registers_dict : dict or None
            Register dictionary per device. Falls back to self attribute.
        register_address : int or str or None
            Specific register address to collect.
        register_name : str or None
            Specific register name to collect.
        lookfor : str or None
            Pattern to search for in register collection.
        flatten : bool
            Whether to flatten the output of device df collection.
        validate : bool
            Whether to validate during device df collection.

        peripheral_registerIDs : dict or None
            {data_key: {localID: register}}. If None, uses self.peripheral_registerIDs.
        return_type : str
            Return type for timestamp collection: 'list', 'dict', or 'both'.

        data_array_name : str or None
            Name for the output DataArray. Defaults to f'{data_key}_data' or 'data_array'.
        da_attrs : dict or None
            Attributes for the DataArray. Falls back to self.data_array_attrs.

        lookup_array_name : str or None
            Name for the lookup DataArray. Defaults to f'{data_key}_lookup' or 'lookup_array'.
        lookup_attrs : dict or None
            Attributes for the lookup array. Falls back to self.lookup_array_attrs.

        fill_value : any or None
            Fill value for updating the DataArray. Falls back to self.fill_value.

        verbose : bool or None
            If True, prints progress. Falls back to self.verbose.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray, dict]
            - data_array: The updated DataArray with values filled in.
            - lookup_da: The lookup matrix DataArray.
            - lookup_virtual_coords: Dict of unique virtual coordinate values per dimension.
        '''
        if verbose is None: verbose = self.verbose

        #=== 1. Resolve virtual_map ========================================
        if virtual_map is None:
            if self.virtual_maps is None:
                raise ValueError("No virtual_map provided and self.virtual_maps is None.")
            if data_key is not None:
                if data_key not in self.virtual_maps:
                    raise KeyError(f"data_key '{data_key}' not in self.virtual_maps. "
                                   f"Available: {list(self.virtual_maps.keys())}")
                virtual_map = self.virtual_maps[data_key]
            else:
                virtual_map = self.virtual_maps

        if verbose: print(f"--- construct_data_outputs: data_key='{data_key}' ---")

        #=== 2. Collect DataFrames =========================================
        if dfs_dict is None:
            if verbose: print("Step 1: Collecting device DataFrames...")
            dfs_dict = self._collect_device_dfs(
                experiment_directory_path=experiment_directory_path,
                harp_device_yaml_path=harp_device_yaml_path,
                device_type=device_type,
                device_list=device_list,
                device_IDs=device_IDs,
                device_registers_dict=device_registers_dict,
                register_address=register_address,
                register_name=register_name,
                lookfor=lookfor,
                flatten=flatten,
                Return=False,
                verbose=verbose,
                validate=validate,
            )

        #=== 3. Collect timestamps =========================================
        if verbose: print("Step 2: Collecting timestamps...")
        timestamps = self._collect_timestamps_nested_dict(
            dfs_dict=dfs_dict,
            data_key=data_key,
            device_list=device_list,
            peripheral_registerIDs=peripheral_registerIDs,
            return_type=return_type,
            verbose=verbose,
        )

        #=== 4. Construct base DataArray ===================================
        if data_array_name is None:
            data_array_name = f'{data_key}_data' if data_key else 'data_array'

        if verbose: print(f"Step 3: Constructing base DataArray '{data_array_name}'...")
        data_array = self._construct_data_array(
            virtual_map=virtual_map,
            times=timestamps,
            name=data_array_name,
            da_attrs=da_attrs,
            verbose=verbose,
        )

        #=== 5. Construct lookup array =====================================
        if lookup_array_name is None:
            lookup_array_name = f'{data_key}_lookup' if data_key else 'lookup_array'

        if verbose: print(f"Step 4: Constructing lookup array '{lookup_array_name}'...")
        lookup_da, lookup_virtual_coords = self._construct_lookup_array(
            virtual_map=virtual_map,
            name=lookup_array_name,
            lookup_attrs=lookup_attrs,
            verbose=verbose,
        )

        #=== 6. Update DataArray with values ===============================
        if verbose: print("Step 5: Updating DataArray with values from DataFrames...")
        data_array = self._update_data_array(
            data_array=data_array,
            virtual_map=virtual_map,
            dfs_dict=dfs_dict,
            fill_value=fill_value,
            verbose=verbose,
        )

        if verbose: print(f"--- construct_data_outputs complete for data_key='{data_key}' ---")

        return data_array, lookup_da, lookup_virtual_coords

    #===========================================================================



    #===========================================================================
    # 10| Master Function to Load and Process Data for All Type Keys
    #===========================================================================

    def construct_all_data_outputs(self,
                                    # --- _collect_device_dfs args ---
                                    dfs_dict: dict | None = None,
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
                                    validate: bool = False,

                                    # --- _collect_timestamps_nested_dict args ---
                                    peripheral_registerIDs: dict[str, dict[str, str]] | None = None,
                                    return_type: str = 'list',

                                    # --- _construct_data_array args ---
                                    da_attrs: dict | None = None,

                                    # --- _construct_lookup_array args ---
                                    lookup_attrs: dict | None = None,

                                    # --- _update_data_array args ---
                                    fill_value: any  = None,

                                    # --- General ---
                                    verbose: bool | None = None,
                                    ) -> dict[str, tuple[xr.DataArray, xr.DataArray, dict]]:
        '''
        Master function to construct data outputs for all data_keys.
        Calls construct_data_outputs for each data_key in self.data_keys.

        Collects DataFrames once and reuses across all data_keys.

        Parameters
        ----------
        dfs_dict : dict or None
            Pre-loaded nested dict {device: {register: DataFrame}}.
            If None, will call _collect_device_dfs() once.
        experiment_directory_path : str or Path or None
            Path to experiment directory.
        harp_device_yaml_path : str or Path or None
            Path to harp device yaml.
        device_type : str or None
            Device type string.
        device_list : list or None
            List of device names.
        device_IDs : dict or None
            Device ID mapping.
        device_registers_dict : dict or None
            Register dictionary per device.
        register_address : int or str or None
            Specific register address to collect.
        register_name : str or None
            Specific register name to collect.
        lookfor : str or None
            Pattern to search for in register collection.
        flatten : bool
            Whether to flatten the output of device df collection.
        validate : bool
            Whether to validate during device df collection.
        peripheral_registerIDs : dict or None
            {data_key: {localID: register}}.
        return_type : str
            Return type for timestamp collection.
        da_attrs : dict or None
            Attributes for DataArrays.
        lookup_attrs : dict or None
            Attributes for lookup arrays.
        fill_value : any or None
            Fill value for updating DataArrays.
        verbose : bool or None
            Falls back to self.verbose.

        Returns
        -------
        dict[str, tuple[xr.DataArray, xr.DataArray, dict]]
            {data_key: (data_array, lookup_da, lookup_virtual_coords), ...}
        '''
        if verbose is None: verbose = self.verbose

        if self.data_keys is None or not self.data_keys:
            raise ValueError("self.data_keys is None or empty. Cannot iterate over data keys.")

        # Collect DataFrames once
        if dfs_dict is None:
            if verbose: print("Collecting device DataFrames (shared across all data_keys)...")
            dfs_dict = self._collect_device_dfs(
                experiment_directory_path=experiment_directory_path,
                harp_device_yaml_path=harp_device_yaml_path,
                device_type=device_type,
                device_list=device_list,
                device_IDs=device_IDs,
                device_registers_dict=device_registers_dict,
                register_address=register_address,
                register_name=register_name,
                lookfor=lookfor,
                flatten=flatten,
                Return=False,
                verbose=verbose,
                validate=validate,
            )

        results = {}
        for dk in self.data_keys:
            if verbose: print(f"\n{'='*60}")
            results[dk] = self.construct_data_outputs(
                data_key=dk,
                dfs_dict=dfs_dict,
                peripheral_registerIDs=peripheral_registerIDs,
                return_type=return_type,
                da_attrs=da_attrs,
                lookup_attrs=lookup_attrs,
                fill_value=fill_value,
                verbose=verbose,
            )

        return results
    #===========================================================================

################################################################################





################################################################################
# Nosepoke
################################################################################
class Nosepoke(MultiDevice):
    '''

    '''
    
    
    def __init__(
        self,
        # Virtual Map-related Parameters
        virtual_maps: dict = None,
        global_coord_name: str = 'peripherals',
        virtual_coord_names: list = ['device', 'register', 'localID'],
        dict_of: str = 'dicts',
        data_keys: list | None = ['Activations', 'LEDs', 'Valves', 'Rewards'],                             # List of data keys to load (list of str). E.g. ['Behavior', 'Wheel',...]
        data_registerIDs: dict[str, dict[str, str]] | None = None, # Dict of data keys to dict of localID to register address (dict of str: dict of str: str). E.g. {'Activations': {'DIPort0': '32', 'DIPort1': '32',...},...}
        # DataFrame-related Parameters
        experiment_directory_path: str | None = None,              # Path to experimental directory (str or Path)
        harp_device_yaml_path: str | None = None,                  # Path to harp_device yaml file (str or Path)
        device_type: str = 'Behavior',                             # Device type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc.
        device_list: list | None = None,                           # List of device names to load (list of str)
        device_IDs: dict | None = None,                            # Dict of device names to IDs (dict of str: str)
        device_registers_dict: dict | None = None,                 # Dict of device names to list of register addresses (dict of str: list of str). dict= {f'{device_list[n]}': [registers],...}
        lookfor: str | None = None,                                # If provided, will select register paths containing this substring if multiple possible matches exist.
        validate: bool = False,                                    # If True, runs validation checks on devices, device IDs, device registers, and register channels. If False, skips validation.
        # DataArray-related Parameters
        data_array_names: list | None = None,                     # Names to use for DataArrays (list). If None, will use f'{selected_type_key}_data'.
        data_array_attrs: dict | None = None,                      # Dict of attributes to add to DataArrays (dict of list: dict). If None, will use default attributes. 
        lookup_array_names: list | None = None,                   # Names to use for lookup arrays (list). If None, will use f'{selected_type_key}_lookup_matrix'.
        lookup_array_attrs: dict | None = None,                    # Dict of attributes to add to lookup arrays (dict of list: dict). If None, will use default attributes. 
        test_values: bool = False,                                 # If True, use test helper values instead of NaNs for channel type DataArrays (bool)
        fill_value = None,                                         # Value to use for missing data when updating channel type DataArrays (e.g., 0, False). If None, will use NaN for floats and False for bools.
        verbose: bool = False,
    ):
        super().__init__(
            virtual_maps=virtual_maps,
            global_coord_name=global_coord_name,
            virtual_coord_names=virtual_coord_names,
            dict_of=dict_of,
            data_keys=data_keys,
            data_registerIDs=data_registerIDs,
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            lookfor=lookfor,
            validate=validate,
            data_array_names=data_array_names,
            data_array_attrs=data_array_attrs,
            lookup_array_names=lookup_array_names,
            lookup_array_attrs=lookup_array_attrs,
            test_values=test_values,
            fill_value=fill_value,
            verbose=verbose
        )





################################################################################






########################################################################################################################
# Wrappers for renamed, moved, or deprecated functions
########################################################################################################################


################################################################################
# BehaviorNosepoke_prev (deprecated)
################################################################################
class BehaviorNosepoke(MultiDevice):
    '''
    Subclass of MultiDevice for managing Behavior Nosepoke data from HARP devices.

    Translates the old nosepoke-specific terminology into the generic MultiDevice 
    framework by building virtual_maps from channel_list, type_keys, 
    channel_type_localIDs, and channel_type_registerIDs.

    Mapping:
        channel_list              ->  global coordinate values (unified names)
        type_keys                 ->  data_keys
        channel_type_registerIDs  ->  data_registerIDs
        channel_ref_dict          ->  virtual_map  {channel: {device, register, localID}}

    Channel assignment is device-major order:
        channel_list[0]  -> device_list[0], localIDs[0]
        channel_list[1]  -> device_list[0], localIDs[1]
        ...
        channel_list[n]  -> device_list[1], localIDs[0]
        etc.

    Usage:
        nosepoke = BehaviorNosepoke(
            experiment_directory_path='/path/to/experiment',
            harp_device_yaml_path='/path/to/device.yml',
            device_list=['Behavior0', 'Behavior1', ..., 'Behavior5'],
            device_registers_dict={...},
            channel_list=['NP_0', 'NP_1', ..., 'NP_17'],
            type_keys=['Activations', 'LED', 'Counts'],
            channel_type_localIDs={
                'Activations': ['DIPort0', 'DIPort1', 'DIPort2'],
                'LED':         ['DOPort0', 'DOPort1', 'DOPort2'],
            },
            channel_type_registerIDs={
                'Activations': {'DIPort0': '32', 'DIPort1': '32', 'DIPort2': '32'},
                'LED':         {'DOPort0': '34', 'DOPort1': '34', 'DOPort2': '34'},
            },
        )

        # Access outputs (same as MultiDevice):
        nosepoke.data_arrays['Activations']
        nosepoke.lookup_arrays['Activations']
        nosepoke.lookup_virtual_coords['Activations']

        # Backwards-compatible access:
        nosepoke.channel_data_arrays['Activations']
        nosepoke.create_channel_ref_dict(type_key='Activations')
    '''

    def __init__(
        self,
        # Nosepoke-specific Parameters (used to build virtual_maps)
        channel_list: list | None = None,                                       # Ordered global channel names. E.g. ['NP_0', 'NP_1', ..., 'NP_17']
        type_keys: list | None = ['Activations', 'LEDs', 'Valves', 'Rewards'], # Data categories to process. Maps to data_keys in MultiDevice.
        channel_type_localIDs: dict[str, list[str]] | None = None,              # Per-type_key ordered list of localIDs. E.g. {'Activations': ['DIPort0', 'DIPort1', 'DIPort2']}
        channel_type_registerIDs: dict[str, dict[str, str]] | None = None,      # Per-type_key mapping of localID -> register address. E.g. {'Activations': {'DIPort0': '32', ...}}
        # Virtual Map-related Parameters
        virtual_maps: dict | None = None,                                       # If provided directly, skips building from nosepoke config. Otherwise built automatically.
        global_coord_name: str = 'peripherals',                                 # Name of the global coordinate dimension in DataArrays.
        virtual_coord_names: list = ['device', 'register', 'localID'],          # Names of virtual coordinate dimensions.
        dict_of: str = 'dicts',                                                 # Format of virtual_map entries. Options: 'dicts' or 'tuples'.
        data_registerIDs: dict[str, dict[str, str]] | None = None,              # If provided directly, overrides channel_type_registerIDs for peripheral_registerIDs inference.
        # DataFrame-related Parameters
        experiment_directory_path: str | None = None,                           # Path to experimental directory (str or Path)
        harp_device_yaml_path: str | None = None,                               # Path to harp_device yaml file (str or Path)
        device_type: str = 'Behavior',                                          # Device type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc.
        device_list: list | None = None,                                        # List of device names to load (list of str)
        device_IDs: dict | None = None,                                         # Dict of device names to IDs (dict of str: str)
        device_registers_dict: dict | None = None,                              # Dict of device names to list of register addresses (dict of str: list of str). dict= {f'{device_list[n]}': [registers],...}
        lookfor: str | None = None,                                             # If provided, will select register paths containing this substring if multiple possible matches exist.
        validate: bool = False,                                                 # If True, runs validation checks on devices, device IDs, device registers, and register channels.
        # DataArray-related Parameters
        data_array_names: list | None = None,                                   # Names to use for DataArrays (list). If None, will use f'{data_key}_data'.
        data_array_attrs: dict | None = None,                                   # Dict of attributes to add to DataArrays (dict of list: dict). If None, will use default attributes.
        lookup_array_names: list | None = None,                                 # Names to use for lookup arrays (list). If None, will use f'{data_key}_lookup'.
        lookup_array_attrs: dict | None = None,                                 # Dict of attributes to add to lookup arrays (dict of list: dict). If None, will use default attributes.
        test_values: bool = False,                                              # If True, use test helper values instead of NaNs for channel type DataArrays (bool)
        fill_value = None,                                                      # Value to use for missing data when updating channel type DataArrays (e.g., 0, False). If None, will use NaN for floats and False for bools.
        verbose: bool = False,                                                  # If True, prints detailed output during processing (bool)
    ):

        ##===== Store Nosepoke-Specific References =========================
        self.channel_list = channel_list
        self.type_keys = type_keys
        self.channel_type_localIDs = channel_type_localIDs
        self.channel_type_registerIDs = channel_type_registerIDs


        ##===== Build virtual_maps from Nosepoke Config ====================
        #  Only build if not provided directly AND all nosepoke config is present.
        #  Channels are assigned in device-major order:
        #    channel_list[i] -> device_list[i // n_locals], localIDs[i % n_locals]

        if virtual_maps is None and all(v is not None for v in [channel_list, type_keys, channel_type_localIDs, channel_type_registerIDs, device_list]):
            virtual_maps = {}
            for tk in type_keys:
                local_ids = channel_type_localIDs[tk]
                reg_map = channel_type_registerIDs[tk]
                n_locals = len(local_ids)

                if len(channel_list) % n_locals != 0:
                    raise ValueError(
                        f"len(channel_list)={len(channel_list)} is not a multiple "
                        f"of len(channel_type_localIDs['{tk}'])={n_locals}."
                    )
                n_devices_needed = len(channel_list) // n_locals
                if n_devices_needed > len(device_list):
                    raise ValueError(
                        f"channel_list requires {n_devices_needed} devices for "
                        f"type_key '{tk}', but only {len(device_list)} provided."
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
                print(f"BehaviorNosepoke: Built virtual_maps for {list(virtual_maps.keys())} "
                      f"from channel_list ({len(channel_list)} channels) x "
                      f"device_list ({len(device_list)} devices).")


        ##===== Resolve data_registerIDs ===================================
        #  If not provided directly, use channel_type_registerIDs (same format).
        if data_registerIDs is None and channel_type_registerIDs is not None:
            data_registerIDs = channel_type_registerIDs


        ##===== Init MultiDevice ===========================================
        super().__init__(
            virtual_maps=virtual_maps,
            global_coord_name=global_coord_name,
            virtual_coord_names=virtual_coord_names,
            dict_of=dict_of,
            data_keys=type_keys,
            data_registerIDs=data_registerIDs,
            experiment_directory_path=experiment_directory_path,
            harp_device_yaml_path=harp_device_yaml_path,
            device_type=device_type,
            device_list=device_list,
            device_IDs=device_IDs,
            device_registers_dict=device_registers_dict,
            lookfor=lookfor,
            validate=validate,
            data_array_names=data_array_names,
            data_array_attrs=data_array_attrs,
            lookup_array_names=lookup_array_names,
            lookup_array_attrs=lookup_array_attrs,
            test_values=test_values,
            fill_value=fill_value,
            verbose=verbose,
        )


    # #===========================================================================
    # # Backwards-Compatible Convenience Properties
    # #===========================================================================

    # @property
    # def channel_data_arrays(self):
    #     '''Alias for self.data_arrays, keyed by type_key.'''
    #     return self.data_arrays

    # @property
    # def channel_lookup_data_arrays(self):
    #     '''Alias for self.lookup_arrays, keyed by type_key.'''
    #     return self.lookup_arrays

    # @property
    # def channel_reference_dicts(self):
    #     '''
    #     Reconstruct old-style channel_ref_dicts from virtual_maps.
    #     Returns {type_key: {channel: (device, localID)}}.
    #     '''
    #     if self.virtual_maps is None:
    #         return None
    #     return {
    #         tk: {
    #             channel: (info['device'], info['localID'])
    #             for channel, info in vmap.items()
    #         }
    #         for tk, vmap in self.virtual_maps.items()
    #     }

    # #===========================================================================
    # # Backwards-Compatible Convenience Methods
    # #===========================================================================

    # def create_channel_ref_dict(self, type_key=None):
    #     '''
    #     Get {channel: (device, localID)} for a given type_key.
    #     Reconstructed from virtual_maps rather than hardcoded.

    #     Parameters:
    #         type_key (str): Must be one of self.type_keys / self.data_keys.

    #     Returns:
    #         dict[str, tuple[str, str]]: {channel_name: (device, localID)}
    #     '''
    #     if type_key is None:
    #         raise ValueError("type_key must be provided.")
    #     refs = self.channel_reference_dicts
    #     if refs is None or type_key not in refs:
    #         raise KeyError(f"type_key '{type_key}' not found in virtual_maps. "
    #                        f"Available: {list(refs.keys()) if refs else 'None'}")
    #     return refs[type_key]


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
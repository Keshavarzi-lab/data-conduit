'''
harp_extender_core.py
--------------------------------

Description: Core functions for extending HARP functionality.

Contents:
--------------------------------
- construct_peripheral_ref_dict
    - constructs dictionary mapping peripheral names to corresponding devices and local_names               (Non-functional: Refactor should make this call func from LookupArrays or MultiDevices and apply decorator)

- construct_device_reader

- collect_device_folders

- collect_registers

- collect_device_dataframes



'''


################################################################################
# Imports
################################################################################

import os
import re
from pathlib import Path

from pyparsing import col
import harp
import pandas as pd
import numpy as np
import xarray as xr
import yaml

################################################################################




################################################################################
# Construct Device Reader
################################################################################

def construct_device_reader(harp_device_yaml_path):
    """
    Create a harp reader for a specific device using the provided YAML configuration file.

    Parameters:
        harp_device_yaml_path (str or Path): Path to the YAML configuration file for the device.

    Returns:
        harp.Reader: A harp reader object for the specified device.
    """
    if not isinstance(harp_device_yaml_path, (str, Path)):
        raise ValueError("harp_device_yaml_path must be a string or Path object.")
    
    if not os.path.exists(harp_device_yaml_path):
        raise FileNotFoundError(f"The specified YAML file does not exist: {harp_device_yaml_path}")
    
    reader = harp.create_reader(f'{harp_device_yaml_path}')
    return reader

################################################################################




################################################################################
# Collect Device Folders
################################################################################

def collect_device_folders(experiment_directory_path, device_type):
    """
    Get a list of device folders in the experiment directory that match the specified device type.

    Parameters:
        experiment_directory_path (str or Path): Path to the experiment directory.
        device_type (str): The type of device to filter folders by (e.g., 'Behavior').

    Returns:
        sorted([f for f in folder_path.glob(f'{device_type}*') if f.is_dir()], key=lambda p: p.name): A list of folder names that match the specified device type.
    """
    # print('Note: This function is identical to get_device_subfolders and may be redundant.')

    folder_path = Path(experiment_directory_path)
    return sorted([f for f in folder_path.glob(f'{device_type}*') if f.is_dir()], key=lambda p: p.name)

################################################################################




################################################################################
# Collect Registers
################################################################################

def collect_registers(harp_device_yaml_path, register_addresses, verbose=False):
    """
    Find register name(s) by address(es) in the schema file.
    Efficiently loads the YAML file once for multiple lookups.
    
    Parameters:
        harp_device_yaml_path: yaml file path
        register_addresses: single int/string OR list of ints/strings
        verbose: If True, prints additional information.
    Returns: 
        Single string (if input was scalar) OR List of strings (if input was list)
    """
    # 1. Determine if input is a list or scalar
    is_scalar = isinstance(register_addresses, (int, str))
    if is_scalar:
        addresses_to_lookup = [int(register_addresses)]
    else:
        addresses_to_lookup = [int(addr) for addr in register_addresses]

    # 2. Load Schema ONCE
    with open(harp_device_yaml_path, 'r') as file:
        schema = yaml.safe_load(file)
    
    # 3. Create a quick lookup map {address: name} from schema
    #    (Handling the special case for address 8/TimestampSeconds)
    addr_to_name = { props['address']: name for name, props in schema.get('registers', {}).items() }
    addr_to_name[8] = 'TimestampSeconds' 

    # 4. Perform Lookups
    results = []
    for addr in addresses_to_lookup:
        name = addr_to_name.get(addr)
        if name:
            results.append(name)
            if verbose: print(f'Found register "{name}" for address {addr}.')
        else:
            if verbose: print(f"Warning: No register found for address {addr}")
            results.append(None) # Or skip, depending on preference

    # 5. Return in format matching input
    if is_scalar:
        return results[0]
    return [r for r in results if r is not None]

################################################################################




################################################################################
# Collect Device Dataframes
################################################################################

def collect_device_dfs(experiment_directory_path: str | Path = None,
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
                         verbose: bool = False,
                         validate: bool= False,
                         ) -> dict | None:
    '''    
    Unified function to collect dataframes for specified register(s) in device folders within an experiment directory.
    
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
    #===== Initialisations =====
    harp_device_reader= construct_device_reader(harp_device_yaml_path)
    device_folders = collect_device_folders(experiment_directory_path, device_type)

    #===== Mode Determination =====
    if flatten:
        if register_address is None and register_name is None or device_registers_dict is not None:
            raise ValueError("When 'flatten' is True, provide either 'register_address' or 'register_name', and do not provide 'device_registers_dict'.")
        
        if register_address is not None and register_name is not None:
            print("Warning: Both 'register_address' and 'register_name' provided. 'register_name' will take precedence.")
        
        if register_address is None and register_name is not None:
            #     # If only name provided, we need to find the address (simple lookup via reader schema or helper)
            #     if target_addr is None:
            #         # Load yaml temporarily to find address from name
            #         with open(harp_device_yaml_path, 'r') as f:
            #             schema = yaml.safe_load(f)
            #         # Reverse lookup: Name -> Address
            #         for r_name, props in schema.get('registers', {}).items():
            #             if r_name == register_name:
            #                 target_addr = props['address']
            #                 break
            #         if target_addr is None and register_name == 'TimestampSeconds':
            #              target_addr = 8
                        
            #         if target_addr is None:
            #             raise ValueError(f"Could not resolve register name '{register_name}' to an address.")
            raise ValueError(f"Register address not found for name {register_name} in schema. This is currently required to obtain correct filepaths.")
        
        if register_address is not None and register_name is None:
            register_name= collect_registers(harp_device_yaml_path, register_address, verbose=verbose)
            if register_name is None:
                raise ValueError(f"Register name not found for address {register_address} in schema.")
        
        if device_registers_dict is not None:
            print('Warning: when flatten is True, device_registers_dict will be ignored and overwritten.')
        device_registers_dict = {f.name: [(register_address)] for f in device_folders}
    
    if not flatten and device_registers_dict is None:
        raise ValueError("When 'flatten' is False, 'device_registers_dict' must be provided by user.")
    



    #===== Validation
    if validate:
        
        print("--- Performing initial validation and setup ---")
        if device_list is None: print("Warning: device_list is None. Skipping validation")
        if device_IDs is None: print("Warning: device_IDs is None. Skipping validation")
        if device_registers_dict is None: print("Warning: device_registers_dict is None. Skipping validation")
        elif device_list is not None and device_IDs is not None and device_registers_dict is not None:
            pass
        print(f'Validation funcs are under refactor and not yet implemented.')
            # validate_experiment_directory(experiment_directory_path, device_list, device_IDs, device_registers_dict, device_type, harp_device_reader, Return=Return)
            
    
    #================
    device_dfs_dict = {}
    if verbose: print("\n--- Processing devices and registers ---")
    
    
    #===== Iterate Through Devices and Registers to Read Data =====
    for device_folder in device_folders:
        device_name = device_folder.name
        if device_name not in device_registers_dict:
            if verbose: print(f"Skipping folder {device_name} as it's not in device_registers_dict.")
            continue

        if verbose: print(f"\nProcessing device: {device_name}")
        device_dfs_dict[device_name] = {}
        
        # Determine the correct path for data files
        potential_subfolder = device_folder / device_name
        device_path = potential_subfolder if potential_subfolder.is_dir() else device_folder

        for register_address in device_registers_dict[device_name]:
            register_name = collect_registers(harp_device_yaml_path, int(register_address))
            if not register_name:
                if verbose: print(f"Warning: No register name found for address {register_address} on device {device_name}. Skipping.")
                continue

            # Find the specific .bin file for the register
            data_files = list(device_path.glob(f'*_{register_address}_*.bin'))
            if not data_files:
                if verbose: print(f"Warning: No .bin file for register {register_name} (address {register_address}) in {device_path}. Skipping.")
                continue
            
            if len(data_files) > 1:
                if verbose: print(f"Warning: Multiple .bin files found for register {register_name}. Using the first one: {data_files[0]}")

            # Read data and create DataFrame
            try:
                data = getattr(harp_device_reader, register_name).read(data_files[0])
                df = pd.DataFrame(data)
                df.index.name = 'Time'
                device_dfs_dict[device_name][register_address] = df
                if verbose: print(f"  - Loaded register {register_name} (address {register_address}) with shape {df.shape}")
            except Exception as e:
                print(f"Error reading register {register_name} for device {device_name}: {e}")
    
    if flatten:
        # {Device: {register: df}}  -->  {Device: df}
        for device, regs in device_dfs_dict.items():
            if len(regs) > 1:
                raise ValueError(f"Multiple registers found for device {device} in flatten mode. Expected only one.")
            device_dfs_dict= {device: next(iter(regs.values())) for device, regs in device_dfs_dict.items()}
    
    
    return device_dfs_dict
        
################################################################################



















################################################################################################################################################################
# Wrappers for renamed, moved, or deprecated functions
################################################################################################################################################################


################################################################################
def create_device_reader(harp_device_yaml_path):
    """
    Wrapper for construct_device_reader to maintain backward compatibility.

    Parameters:
        harp_device_yaml_path (str or Path): Path to the YAML configuration file for the device.
    Returns:
        harp.Reader: A harp reader object for the specified device.
    """
    print("Warning: 'create_device_reader' is deprecated. Please use 'construct_device_reader' instead.")
    return construct_device_reader(harp_device_yaml_path)

##--------------------------------------------------------------------------------   Original
# def create_device_reader(harp_device_yaml_path):
#     """
#     Create a harp reader for a specific device using the provided YAML configuration file.

#     Parameters:
#         harp_device_yaml_path (str or Path): Path to the YAML configuration file for the device.

#     Returns:
#         harp.Reader: A harp reader object for the specified device.
#     """
#     if not isinstance(harp_device_yaml_path, (str, Path)):
#         raise ValueError("harp_device_yaml_path must be a string or Path object.")
    
#     if not os.path.exists(harp_device_yaml_path):
#         raise FileNotFoundError(f"The specified YAML file does not exist: {harp_device_yaml_path}")
    
#     reader = harp.create_reader(f'{harp_device_yaml_path}')
#     return reader
################################################################################



################################################################################
def get_device_folders(experiment_directory_path, device_type):
    """
    Wrapper for collect_device_folders to maintain backward compatibility.

    Parameters:
        experiment_directory_path (str or Path): Path to the experiment directory.
        device_type (str): The type of device to filter folders by (e.g., 'Behavior').
    Returns:
        sorted([f for f in folder_path.glob(f'{device_type}*') if f.is_dir()], key=lambda p: p.name): A list of folder names that match the specified device type.
    """
    print("Warning: 'get_device_folders' is deprecated. Please use 'collect_device_folders' instead.")
    return collect_device_folders(experiment_directory_path, device_type)

def get_device_subfolders(experiment_directory_path, device_type):
    """
    Wrapper for collect_device_folders to maintain backward compatibility.

    Parameters:
        experiment_directory_path (str or Path): Path to the experiment directory.
        device_type (str): The type of device to filter folders by (e.g., 'Behavior').
    Returns:
        sorted([f for f in folder_path.glob(f'{device_type}*') if f.is_dir()], key=lambda p: p.name): A list of folder names that match the specified device type.
    """
    print("Warning: 'get_device_subfolders' is deprecated. Please use 'collect_device_folders' instead.")
    return collect_device_folders(experiment_directory_path, device_type)
# #--------------------------------------------------------------------------------   Original
# def get_device_folders(experiment_directory_path, device_type):
#     """
#     Get a list of device folders in the experiment directory that match the specified device type.

#     Parameters:
#         experiment_directory_path (str or Path): Path to the experiment directory.
#         device_type (str): The type of device to filter folders by (e.g., 'Behavior').

#     Returns:
#         sorted([f for f in folder_path.glob(f'{device_type}*') if f.is_dir()], key=lambda p: p.name): A list of folder names that match the specified device type.
#     """
#     # print('Note: This function is identical to get_device_subfolders and may be redundant.')

#     folder_path = Path(experiment_directory_path)
#     return sorted([f for f in folder_path.glob(f'{device_type}*') if f.is_dir()], key=lambda p: p.name)

# def get_device_subfolders(experiment_directory_path, device_type):
    """
    Get a list of subfolders in the experiment directory that match the specified device type.

    Parameters:
        experiment_directory_path (str or Path): Path to the experiment directory.
        device_type (str): The type of device to filter folders by (e.g., 'Behavior').

    Returns:
        sorted([f for f in folder_path.glob(f'{device_type}*') if f.is_dir()], key=lambda p: p.name): A list of sorted subfolder names that match the specified device type.
    """
    # print('Note: This function is identical to get_device_folders and may be redundant.')

    folder_path = Path(experiment_directory_path)
    return sorted([f for f in folder_path.glob(f'{device_type}*') if f.is_dir()], key=lambda p: p.name)
################################################################################



################################################################################

def get_register_name_by_address(harp_device_yaml_path, register_address, verbose=False):
    print("Warning: 'get_register_name_by_address' is deprecated. Please use 'collect_registers' instead.")
    return collect_registers(harp_device_yaml_path, register_address, verbose)
##--------------------------------------------------------------------------------   Original
# def get_register_name_by_address(harp_device_yaml_path, register_address, verbose=False):
    """
    Find the register name by its address in the schema file.
    
    Parameters:
        harp_device_yaml_path: yaml file path
        target_address: register address number
        verbose: If True, prints additional information.
    Returns: 
        Register name if found, else None.
    """
    with open(harp_device_yaml_path, 'r') as file:
        schema = yaml.safe_load(file)
    

    for name, properties in schema.get('registers', {}).items():
        if properties.get('address') == register_address:
            if verbose:
                print(f'Found register name "{name}" for address {register_address}.')
            return name
        elif register_address == 8:
            if verbose:
                print('Special case: Register address 8 corresponds to TimestampSeconds. See Harp Core YAML for details.')
            return 'TimestampSeconds'
    
    return None
################################################################################


################################################################################

def get_register_names_by_addresses(harp_device_yaml_path, register_addresses, verbose=False):
    print("Warning: 'get_register_names_by_addresses' is deprecated. Please use 'collect_registers' instead.")
    return collect_registers(harp_device_yaml_path, register_addresses, verbose)
##--------------------------------------------------------------------------------   Original
# def get_register_names_by_addresses(harp_device_yaml_path, register_addresses, verbose=False):
    # """
    # Find register names by their addresses in the schema file.
    
    # Parameters:
    #     harp_device_yaml_path: yaml file path
    #     register_addresses: list of register address numbers (ints or numeric strings)
    #     verbose (bool, optional): If True, prints additional information.
    # Returns: 
    #     List of register names found.
    # """
    # names = []
    # for addr in register_addresses:
    #     # ensure we pass an integer to get_register_name_by_address
    #     try:
    #         addr_int = int(addr)
    #     except Exception:
    #         if verbose:
    #             print(f"Warning: invalid register address '{addr}' - must be integer-like. Skipping.")
    #         continue

    #     name = get_register_name_by_address(harp_device_yaml_path, addr_int, verbose=verbose)
    #     if name:
    #         names.append(name)
    #     else:
    #         if verbose:
    #             print(f"Warning: No register found for address {addr_int}")
        
    #     # if verbose:
    #     #     print(f'Found register name "{name}" for address {addr_int}.')
    
    # return names
################################################################################

################################################################################
def get_devices_register_dfs_dict(experiment_directory_path: str | Path = None,
                         harp_device_yaml_path: str | Path = None,
                         device_type: str = None,
                         device_list: list | None = None,
                         device_IDs: dict | None = None,
                         device_registers_dict: dict | None = None,
                         register_address: int | str = None,
                         register_name: str = None,
                         lookfor: str = None,
                         flatten: bool = True,
                         Return: bool = False,
                         verbose: bool = False,
                         validate: bool= False,
                         ) -> dict | None:
    print("Warning: 'get_devices_register_dfs_dict' is deprecated. Please use 'collect_device_dfs' instead.")
    return collect_device_dfs(experiment_directory_path=experiment_directory_path,
                         harp_device_yaml_path=harp_device_yaml_path,
                         device_type=device_type,
                         device_list=device_list,
                         device_IDs=device_IDs,
                         device_registers_dict=device_registers_dict,
                         register_address=register_address,
                         register_name=register_name,
                         lookfor=lookfor,
                         flatten=flatten,
                         Return=Return,
                         verbose=verbose,
                         validate=validate,
                         )
##--------------------------------------------------------------------------------   Original
# def get_devices_register_dfs_dict(experiment_directory_path= None, harp_device_yaml_path= None, device_type= None, device_list= None, device_IDs=None, register_address= None, register_name=None, verbose= False):
    """
    Create a dictionary containing the DataFrame for a specified register for each device folder.

    Parameters:
        experiment_directory_path (str or Path): Path to the directory containing experiment logs.
        harp_device_yaml_path (str or Path): Path to the YAML configuration file for the device.
        device_type (str): The type of device to filter folders by (e.g., 'Behavior').
        register_address (int or str): The address of the register to read.
        register_name (str): The name of the register to read. If provided, takes precedence over register_address.
        device_list (list of str): List of expected device names.
        device_IDs (dict): Dictionary mapping device names to their expected IDs.
        verbose (bool): If True, prints additional information during processing including validation steps.


    Returns:
        dict: A dictionary where keys are device folder names and values are DataFrames of the specified register.
    """

 #===== Get Subfolders for Device Type =====

    devicetype_folders = get_device_subfolders(experiment_directory_path, device_type)



    #===== Validate Register Name or Address =====

    if register_address is None and register_name is None:
        raise ValueError("Either register_address or register_name must be provided.")
    
    if register_address is not None and register_name is not None:
        address_matched_name= get_register_name_by_address(harp_device_yaml_path, int(register_address))
        if address_matched_name != register_name:
            raise ValueError(f"Provided register_name '{register_name}' does not match the name '{address_matched_name}' for address {register_address}. It is recommended to provide only one of register_address or register_name.")
        else:
            if verbose:
                print(f'Both register_address ({register_address}) and register_name ({register_name}) provided and are matching. Proceeding with register_name.')

    if register_name is None and register_address is not None:
        register_name = get_register_name_by_address(harp_device_yaml_path, int(register_address))
        if register_name is None:
            raise ValueError(f"No register found for address {register_address}")
        if verbose:
            print(f"Register name for address {register_address} is '{register_name}'")

    if register_name is not None and register_address is None:
        if verbose:
            print(f"Using provided register_name '{register_name}'")

    #===== Get Device Reader =====
    harp_device_reader = create_device_reader(harp_device_yaml_path)

    if verbose:
        #===== Validate Devices Present =====
        print(f'\n\n-----------------------------------------------------\nValidating devices present...\n-----------------------------------------------------')
        validate_devices_present(experiment_directory_path, device_list, device_type, Return=False)
        print(f'\n\n-----------------------------------------------------\n')

        #==== Validate Device IDs =====
        print(f'\n\n-----------------------------------------------------\nValidating device IDs...\n-----------------------------------------------------')
        validate_device_IDs(experiment_directory_path, device_IDs, device_type)
        print(f'\n\n-----------------------------------------------------\n')

        #==== Validate Device Registers =====
        print(f'\n\n-----------------------------------------------------\nValidating device registers...\n-----------------------------------------------------')
        validate_device_registers(experiment_directory_path=experiment_directory_path, device_registers=device_registers_dict, Return=False)
        print(f'\n\n-----------------------------------------------------\n')

        #==== Validate Register Channels =====
        print(f'\n\n-----------------------------------------------------\nValidating register channels...\n-----------------------------------------------------')
        validate_register_channels(experiment_directory_path=experiment_directory_path, device_registers=device_registers_dict, Return=False)
        print(f'\n\n-----------------------------------------------------\n')

    #===== Read Register Data for Each Device Folder =====
    device_register_dfs_dict = {}                           # Dictionary to hold DataFrames for each device
    for i, devicetype_folder in enumerate(devicetype_folders):
        device_name = devicetype_folder.name

        # If a subfolder matching the device_name exists within devicetype_folder, use it; otherwise use the folder itself
        potential_subfolder = devicetype_folder / device_name
        if potential_subfolder.exists():
            device_path = potential_subfolder  # use subfolder if it exists
        else:
            device_path = devicetype_folder  # use the folder itself
        if verbose:
            print(f'\nDevice path is: {device_path}')


        if not device_path.exists():
            if verbose:
                print(f'Warning: Device path does not exist: {device_path}. Skipping this device.')
            continue

        if verbose:
            print(f'Processing device folder: {device_name} ({i} of {len(devicetype_folders)})')

        if device_list is not None:
            if device_name not in device_list:
                if verbose:
                    print(f'Note: {device_name} does not appear in device_list')
            else:
                if verbose:
                    print(f'Note: {device_name} appears in device_list as entry {device_list.index(device_name)}')

        #===== Get .bin Files for the Specified Register =====
        data_file = list(device_path.glob(f'{device_name}_{register_address}_*.bin'))
        if not data_file:
            if verbose:
                print(f'Warning: No .bin file found for register {register_name} (address {register_address}) in {device_path}. Skipping this device.')
            continue
        if len(data_file) > 1:
            if verbose:
                print(f'Warning: Multiple .bin files found for register {register_name} (address {register_address}) in {device_path}. Using the first one found.')

        #===== Read the Data from the .bin File =====
        data= getattr(harp_device_reader, register_name).read(data_file[0])

        data_df = pd.DataFrame(data)
        data_df.index.name= 'Time'

        device_register_dfs_dict[device_name] = data_df
        if verbose:
            print(f'Added DataFrame for {device_name} with shape {data_df.shape} to the dictionary.\n')

    return device_register_dfs_dict

################################################################################


################################################################################
def get_dict_of_devices_registers_dfs_dicts(experiment_directory_path: str | Path = None,
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
                         verbose: bool = False,
                         validate: bool= False,
                         ) -> dict | None:
    print("Warning: 'get_dict_of_devices_registers_dfs_dicts' is deprecated. Please use 'collect_device_dfs' instead.")
    return collect_device_dfs(experiment_directory_path=experiment_directory_path,
                         harp_device_yaml_path=harp_device_yaml_path,
                         device_type=device_type,
                            device_list=device_list,
                            device_IDs=device_IDs,
                            device_registers_dict=device_registers_dict,
                            register_address=register_address,
                            register_name=register_name,
                            lookfor=lookfor,
                            flatten=flatten,
                            Return=Return,
                            verbose=verbose,
                            validate=validate,
                            )
##--------------------------------------------------------------------------------   Original
# def get_dict_of_devices_register_dfs_dicts(experiment_directory_path=None, harp_device_yaml_path=None, device_type=None, device_list=None, device_IDs=None, device_registers_dict=None, Return= False, verbose=False):
    '''
    Create a nested dictionary containing DataFrames for all specified registers for each device folder.
    This optimised version performs validations only once, then reads all register data.
    -----------------------------------

    Parameters:
        experiment_directory_path (str or Path): Path to the directory containing experiment logs.
        harp_device_yaml_path (str or Path): Path to the YAML configuration file for the device.
        device_type (str): The type of device to filter folders by (e.g., 'Behavior').
        device_list (list of str): List of expected device names.
        device_IDs (dict): Dictionary mapping device names to their expected IDs.
        device_registers_dict (dict): Dictionary mapping device names to their expected register IDs.
        Return (bool): If True, returns the nested dictionary. If False, only prints progress.
        verbose (bool): If True, prints additional information during processing.

    Returns:
        dict: A nested dictionary where keys are device folder names and values are dictionaries of DataFrames for each register. Structure dict[device_name][register_address][localID]

    '''
    if not device_registers_dict:
        print("Warning: device_registers_dict is empty or None.")
        return {}

    # ===== Perform Validations and Setup Once =====
    harp_device_reader = create_device_reader(harp_device_yaml_path)
    if verbose:
        print("--- Performing initial validation and setup ---")
        if device_list is None:
            print("Warning: device_list is None. Skipping validation")
        if device_IDs is None:
            print("Warning: device_IDs is None. Skipping validation")
        if device_registers_dict is None:
            print("Warning: device_registers_dict is None. Skipping validation")
        elif device_list is not None and device_IDs is not None and device_registers_dict is not None:
            validate_experiment_directory(experiment_directory_path, device_list, device_IDs, device_registers_dict, device_type, harp_device_reader, Return=Return)

    devicetype_folders = get_device_subfolders(experiment_directory_path, device_type)

    dict_of_devices_registers_dfs_dicts = {}
    if verbose:
        print("\n--- Processing devices and registers ---")


    # ===== Iterate Through Devices and Registers to Read Data =====
    for device_folder in devicetype_folders:
        device_name = device_folder.name
        if device_name not in device_registers_dict:
            if verbose:
                print(f"Skipping folder {device_name} as it's not in device_registers_dict.")
            continue

        if verbose:
            print(f"\nProcessing device: {device_name}")
        dict_of_devices_registers_dfs_dicts[device_name] = {}
        
        # Determine the correct path for data files
        potential_subfolder = device_folder / device_name
        device_path = potential_subfolder if potential_subfolder.is_dir() else device_folder

        for register_address in device_registers_dict[device_name]:
            register_name = get_register_name_by_address(harp_device_yaml_path, int(register_address))
            if not register_name:
                if verbose:
                    print(f"Warning: No register name found for address {register_address} on device {device_name}. Skipping.")
                continue

            # Find the specific .bin file for the register
            data_files = list(device_path.glob(f'*_{register_address}_*.bin'))
            if not data_files:
                if verbose:
                    print(f"Warning: No .bin file for register {register_name} (address {register_address}) in {device_path}. Skipping.")
                continue
            
            if len(data_files) > 1:
                if verbose:
                    print(f"Warning: Multiple .bin files found for register {register_name}. Using the first one: {data_files[0]}")

            # Read data and create DataFrame
            try:
                data = getattr(harp_device_reader, register_name).read(data_files[0])
                df = pd.DataFrame(data)
                df.index.name = 'Time'
                dict_of_devices_registers_dfs_dicts[device_name][register_address] = df
                if verbose:
                    print(f"  - Loaded register {register_name} (address {register_address}) with shape {df.shape}")
            except Exception as e:
                print(f"Error reading register {register_name} for device {device_name}: {e}")

    return dict_of_devices_registers_dfs_dicts
################################################################################
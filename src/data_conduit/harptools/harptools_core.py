'''
HarpTools Module.
--------------------------------

Description: 
    Core functions for extending HARP functionality. 
    Provides utilities for device reader construction, register collection,
    and reader factory used in the data conduit IO module.

Contents:
--------------------------------
- construct_device_reader
    Create a harp reader for a specific device using the provided 
    YAML configuration file.
- collect_registers
    Find register name(s) by address(es) in the schema file. 
    Efficiently loads the YAML file once for multiple lookups.
- read_harp_bin
    Read a HARP .bin file into a pandas DataFrame. Parses the register
    address from the filename, resolves it to a register name using
    collect_registers, and reads using a cached device reader from
    construct_device_reader.


'''


################################################################################
# Imports
################################################################################



import os # noqa
from pathlib import Path

import yaml
import pandas as pd
import harp 


################################################################################






################################################################################
# Construct Device Reader
################################################################################

def construct_device_reader(
        harp_device_yaml_path: str | Path
        ):
    """
    Create a harp reader for a specific device using the provided YAML configuration file.

    Parameters
    ----------
    harp_device_yaml_path : str | Path
        Path to the YAML configuration file for the device.

    Returns
    -------
    harp.Reader
        A harp reader object for the specified device.
    """
    if not isinstance(harp_device_yaml_path, str | Path):
        raise TypeError(
            f"harp_device_yaml_path must be a str or Path, got {type(harp_device_yaml_path)}"
        )

    if not os.path.exists(harp_device_yaml_path):
        raise FileNotFoundError(
            f"The specified YAML file does not exist: {harp_device_yaml_path}"
        )

    return harp.create_reader(str(harp_device_yaml_path))

################################################################################







################################################################################
# Collect Registers
################################################################################

def collect_registers(harp_device_yaml_path: str | Path, 
                      register_addresses: int | str | list[int] | list[str], 
                      verbose: bool = False
                      ) -> str | list[str] | None:
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
    is_scalar = isinstance(register_addresses, (int, str))          # noqa
    if is_scalar:                                                   # noqa | Ignored SIM108 to improve readability.
        addresses_to_lookup = [int(register_addresses)]
    else:
        addresses_to_lookup = [int(addr) for addr in register_addresses]    

    # 2. Load Schema (once for all lookups)
    with open(harp_device_yaml_path) as file:
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
            if verbose: 
                print(f'Found register "{name}" for address {addr}.')
        else:
            if verbose: 
                print(f"Warning: No register found for address {addr}")
            results.append(None) # Or skip, depending on preference

    # 5. Return in format matching input
    if is_scalar:
        return results[0]
    return [r for r in results if r is not None]

################################################################################







################################################################################
# Read HARP Binary
################################################################################

_harp_reader_cache = {}

def read_harp_bin(path: str | Path,
                  harp_device_yaml_path: str | Path = None,
                  **kwargs,
                  ) -> pd.DataFrame:
    '''
    Read a HARP .bin file into a pandas DataFrame.

    Parses the register address from the filename, resolves it to a
    register name using collect_registers, and reads using a cached 
    device reader from construct_device_reader.

    Parameters
    ----------
    path : str or Path
        Path to the .bin file.
        Expected filename format: {DeviceName}_{RegisterAddress}_{Timestamp}.bin
    harp_device_yaml_path : str or Path
        Path to the YAML configuration file for the device.

    Returns
    -------
    pd.DataFrame
        DataFrame with Time as the index.
    '''
    if harp_device_yaml_path is None:
        raise ValueError("harp_device_yaml_path is required.")

    yaml_key = str(harp_device_yaml_path)
    if yaml_key not in _harp_reader_cache:
        _harp_reader_cache[yaml_key] = construct_device_reader(harp_device_yaml_path)

    reader = _harp_reader_cache[yaml_key]
    register_address = int(Path(path).stem.split('_')[1])
    register_name = collect_registers(harp_device_yaml_path, register_address)

    if register_name is None:
        raise ValueError(
            f"No register found for address {register_address} "
            f"(parsed from '{Path(path).name}')."
        )

    data = getattr(reader, register_name).read(path)
    df = pd.DataFrame(data)
    df.index.name = 'Time'
    return df

################################################################################

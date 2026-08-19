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

- collect_harp_dfs
    HARP-flavoured wrapper around collect_dfs. Walks the experiment
    directory for .bin files under device_type folders, reads them using
    read_harp_bin, then applies HARP-specific cleanup:
    1. Collapse Bonsai double-folder structure (Device/Device/files → Device/files), 
        if present.
    2. Rekey long file stems to register address strings (if rekey=True)
        so the device tree becomes {device: {register_address: df}}.
    3. Optionally filter to only expected registers.
    4. Optionally warn about missing expected devices/registers.

- HARP Post-Processing Helpers
    Internal functions for cleaning up the nested dictionary structure
    returned by collect_harp_dfs, such as collapsing double folders and
    rekeying file stems to register addresses.
    
    - _collapse_double_folders:
        Collapse Bonsai double-folder structure (Device/Device/files → Device/files), if present.
    
    - _rekey_to_register_addresses:
        Rekey long file stems to register address strings (if rekey=True) so the device tree becomes 
        {device: {register_address: df}}. 
        If a stem cannot be parsed (e.g. no underscore), or if two stems would map to the same register 
        key, the original stem is kept as-is and a warning is printed.

    - _filter_device_registers:
        Filter the device tree to only include expected registers.
    
    - _warn_missing_expected_devices:
        Warn about any expected devices that are missing from the device tree.
    
    - _warn_missing_expected_registers:
        Warn about any expected registers that are missing from the device tree.


'''


################################################################################
# Imports
################################################################################



import os # noqa
from pathlib import Path

import yaml
import pandas as pd
import harp 

from data_conduit.io import collect_dfs
from data_conduit.utils import starts_with


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




################################################################################
# HARP Post-Processing Helpers
################################################################################


def _collapse_double_folders(dfs_dict: dict) -> dict:
    """
    Collapse duplicated device folder levels in loaded HARP trees.

    Bonsai often creates {DeviceName: {DeviceName: {files...}}} structures.
    This collapses them to {DeviceName: {files...}}.

    Parameters
    ----------
    dfs_dict : dict
        Nested dictionary from collect_dfs.

    Returns
    -------
    dict
        Dictionary with redundant nesting removed.
    """
    collapsed = {}
    for device_name, value in dfs_dict.items():
        if isinstance(value, dict) and list(value.keys()) == [device_name]:
            collapsed[device_name] = value[device_name]
        else:
            collapsed[device_name] = value
    return collapsed


def _rekey_to_register_addresses(dfs_dict: dict, verbose: bool = False) -> dict:
    """
    Rekey HARP file stems to register address strings.

    Converts keys like ``Behavior0_32_1904-01-01T02-00-00`` into ``32``
    so the device tree becomes ``{device: {register_address: df}}``.

    If a stem cannot be parsed (e.g. no underscore), or if two stems
    would map to the same register key, the original stem is kept as-is
    and a warning is printed.

    Parameters
    ----------
    dfs_dict : dict
        Nested dict where second-level keys are raw file stems.
    verbose : bool
        If True, print warnings when keys cannot be rekeyed.

    Returns
    -------
    dict
        Same structure with second-level keys replaced by register address
        strings where possible, original stems kept where not.
    """
    rekeyed = {}
    for device_name, value in dfs_dict.items():
        if not isinstance(value, dict):
            rekeyed[device_name] = value
            continue

        register_dict = {}
        for key, item in value.items():
            parts = key.split('_')
            if len(parts) > 1:
                register_key = parts[1]
            else:
                if verbose:
                    print(f"Warning: Cannot parse register address from '{key}' "
                          f"for device '{device_name}'. Keeping original key.")
                register_key = key

            if register_key in register_dict:
                if verbose:
                    print(f"Warning: Register key '{register_key}' already exists "
                          f"for device '{device_name}'. Keeping original key '{key}'.")
                register_dict[key] = item
            else:
                register_dict[register_key] = item

        rekeyed[device_name] = register_dict

    return rekeyed


def _filter_device_registers(
        dfs_dict: dict,
        device_registers: dict | None,
) -> dict:
    """
    Keep only requested registers for devices listed in device_registers.

    Parameters
    ----------
    dfs_dict : dict
        Nested dict ``{device: {register: df}}``.
    device_registers : dict or None
        ``{device_name: [register_addresses]}``. If None, no filtering.

    Returns
    -------
    dict
        Filtered dictionary.
    """
    if device_registers is None:
        return dfs_dict

    filtered = {}
    for device_name, registers in dfs_dict.items():
        if device_name not in device_registers:
            filtered[device_name] = registers
            continue

        allowed = {str(r) for r in device_registers[device_name]}
        if isinstance(registers, dict):
            filtered[device_name] = {
                reg_name: item
                for reg_name, item in registers.items()
                if reg_name in allowed
            }
        else:
            filtered[device_name] = registers

    return filtered


def _warn_missing_expected_devices(
        dfs_dict: dict,
        device_list: list[str] | None,
        verbose: bool,
) -> None:
    """Print warnings for expected devices not found in loaded data."""
    if not verbose or not device_list:
        return
    for device_name in device_list:
        if device_name not in dfs_dict:
            print(f"Warning: expected device '{device_name}' was not found in loaded data.")


def _warn_missing_expected_registers(
        dfs_dict: dict,
        device_registers: dict | None,
        verbose: bool,
) -> None:
    """Print warnings for expected registers not found in loaded data."""
    if not verbose or not device_registers:
        return
    for device_name, expected in device_registers.items():
        if device_name not in dfs_dict:
            print(f"Warning: expected device '{device_name}' was not found in loaded data.")
            continue
        loaded = dfs_dict[device_name]
        if not isinstance(loaded, dict):
            continue
        for register in expected:
            if str(register) not in loaded:
                print(f"Warning: expected register '{register}' not found for device '{device_name}'.")


################################################################################




################################################################################
# Collect HARP DataFrames
################################################################################

def collect_harp_dfs(
        base_path: str | Path,
        harp_device_yaml_path: str | Path,
        device_type: str = 'Behavior',
        device_registers: dict | None = None,
        device_list: list[str] | None = None,
        rekey: bool = True,
        keep_empty: bool = False,
        verbose: bool = False,
        **kwargs,
) -> dict:
    '''
    HARP-flavoured wrapper around collect_dfs.

    Walks the experiment directory for .bin files under device_type folders,
    reads them using read_harp_bin, then applies HARP-specific cleanup:
    1. Collapse Bonsai double-folder structure (Device/Device/files → Device/files)
    2. Rekey ugly file stems to register address strings (if rekey=True)
    3. Optionally filter to only expected registers
    4. Optionally warn about missing expected devices/registers

    Parameters
    ----------
    base_path : str or Path
        Root experiment directory.
    harp_device_yaml_path : str or Path
        Path to the HARP device YAML schema.
    device_type : str
        Folder prefix to select (e.g. 'Behavior', 'SoundCard').
    device_registers : dict or None
        ``{device_name: [register_addresses]}`` to filter and validate.
        If None, all registers found are kept.
    device_list : list[str] or None
        Expected device names. Used only for warning messages.
    rekey : bool
        If True (default), rekey file stems to register address strings.
        If False, keep the original file stems as dict keys.
    keep_empty : bool
        If True, preserve empty subdirectories as empty dicts.
    verbose : bool
        If True, print warnings for missing devices/registers.
    **kwargs
        Additional level selectors passed to collect_dfs
        (e.g. l1_selector, l2_selector).

    Returns
    -------
    dict
        Clean nested dictionary: ``{device_name: {register_address: pd.DataFrame}}``.
        Register address keys are strings (e.g. '32', '34') if rekey=True,
        or raw file stems if rekey=False.

    Example
    -------
    >>> dfs = collect_harp_dfs(
    ...     base_path='./experiment_2025/',
    ...     harp_device_yaml_path='./device.yml',
    ...     device_type='Behavior',
    ...     device_registers={'Behavior0': ['32', '34'], 'Behavior1': ['32', '34']},
    ... )
    >>> dfs['Behavior0']['32']  # DataFrame with Time index
    '''

    # 1. Generic walk + read
    dfs_dict = collect_dfs(
        base_path=base_path,
        readers={'.bin': read_harp_bin},
        reader_kwargs={'.bin': {'harp_device_yaml_path': harp_device_yaml_path}},
        l0_selector=starts_with(device_type),
        keep_empty=keep_empty,
        verbose=verbose,
        **kwargs,
    )

    # 2. HARP-specific cleanup
    dfs_dict = _collapse_double_folders(dfs_dict)
    if rekey:
        dfs_dict = _rekey_to_register_addresses(dfs_dict, verbose=verbose)

    # 3. Filter and validate
    dfs_dict = _filter_device_registers(dfs_dict, device_registers)
    _warn_missing_expected_devices(dfs_dict, device_list, verbose)
    _warn_missing_expected_registers(dfs_dict, device_registers, verbose)

    return dfs_dict

################################################################################

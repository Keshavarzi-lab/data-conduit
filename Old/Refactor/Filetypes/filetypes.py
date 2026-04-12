'''
Utilities for managing and loading various filetypes from Bonsai which are not HARP binary data


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
def _collect_filetype_path(folder_path: str | Path = None,
                           file_type: str = None,
                           return_type: str = 'first',              # Options: 'first', 'all', 'count'
                           subfolder_index: int = 0,                # If folder_path contains multiple subfolders, specify which one to use (int). Default is first subfolder (0).
                           verbose: bool = False,
                           ) -> Path | list[Path] | None:
    '''
    Collect filepaths of specified type from an experimental directory.
    -----------------------------------
    Parameters:
        folder_path: str or Path
            Path to the subfolder within the experimental directory.
        file_type: str
            File type to search for (e.g., 'csv', 'YAML', 'JSONL').
        return_type: str, optional
            Type of return value. Options are:
                - 'first': Return the first matching file path or None if no files found.
                - 'all': Return a list of all matching file paths.
                - 'count': Return the count of matching files.
            Default is 'first'.
        subfolder_index: int, optional
            If folder_path contains multiple subfolders, specify which one to use (int). Default is first subfolder (0).
        verbose: bool, optional
            If True, prints detailed output during the search process. Default is False.
    '''
    
    if not folder_path:     raise ValueError("Folder path must be provided.")
    if verbose:             print(f"Searching for {file_type} files in {folder_path}")

    
    '''
    Collect file paths of the specified type. Use rglob to search recursively in subfolders.
    Sorted to ensure consistent order, especially when using 'first' return type. 
    Could otherwise return different files on different runs if the filesystem order changes.
    '''
    
    file_paths = sorted(Path(folder_path).rglob(f"*.{file_type}")) if file_type else []

    if verbose:                     print(f'Searching for *.{file_type} files in {folder_path} found {len(file_paths)} files.')

    if return_type == 'first':      return file_paths[subfolder_index] if file_paths else None
    elif return_type == 'all':      return file_paths
    elif return_type == 'count':    return len(file_paths)
    else:                           raise ValueError(f"Invalid return_type: {return_type}. Must be 'first', 'all', or 'count'.")



#===============================================================================





#===============================================================================
# 2| CSV to Dataframe
#===============================================================================
def _CSV_to_df(filepath: str | Path,
               csv_kwargs: dict | None = None,
                verbose: bool = False
                ) -> pd.DataFrame:
        
    '''
    Loads CSV data from a file into a pandas DataFrame.
    -----------------------------------
    Parameters:
        filepath: str or Path
            Path to the CSV file.
        csv_kwargs: dict or None, optional
            Optional dictionary of keyword arguments to pass to pd.read_csv. If None, default arguments will be used.
        verbose: bool, optional
            If True, prints detailed output during loading. Default is False.
    Returns:
        pd.DataFrame: 
            DataFrame containing the loaded CSV data.
    '''
    if not isinstance(filepath, (str, Path)):
        raise ValueError("filepath must be a string or Path object.")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The specified CSV file does not exist: {filepath}")
    if csv_kwargs is None:
        csv_kwargs= {'header': 0,
                     'dtype': str,
                     'engine': 'python',
                     'sep': r',(?![^{]*})',  # Split on commas not inside braces
                     'index_col': 0,
                     }
        if verbose:
            print("Using default csv_kwargs: ", csv_kwargs)
    else:
        if verbose:
            print("Using provided csv_kwargs: ", csv_kwargs)
    df= pd.read_csv(filepath, **csv_kwargs)

    if verbose:
        print("Loaded DataFrame shape: ", df.shape)

    return df



#===============================================================================



# TODO: Check if custom kwargs are actually working here
#===============================================================================
# 3| JSONL to DataFrame
#===============================================================================

def _JSONL_to_df(filepath: str | Path,
                 jsonl_kwargs: dict | None = None,
                 verbose: bool = False
                 ) -> pd.DataFrame:
    '''
    Load JSONL data from a file into a pandas DataFrame.
    -----------------------------------
    Parameters:
        filepath (str or Path): Path to the JSONL file.
        jsonl_kwargs (dict or None): Optional dictionary of keyword arguments to pass to pd.read_json. If None, default arguments will be used.
        verbose (bool): If True, prints detailed output during loading.
    Returns:
        pd.DataFrame: DataFrame containing the loaded JSONL data.
    '''
    if not isinstance(filepath, (str, Path)):
        raise ValueError("filepath must be a string or Path object.")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The specified JSONL file does not exist: {filepath}")
    if jsonl_kwargs is None:
        jsonl_kwargs= {'lines': True,
                       'dtype': str,
                       'orient': 'records',
                       'convert_dates': False,
                       'precise_float': True,
                       }
        if verbose:
            print("Using default jsonl_kwargs: ", jsonl_kwargs)
    else:
        if verbose:
            print("Using provided jsonl_kwargs: ", jsonl_kwargs)
    df= pd.read_json(filepath, **jsonl_kwargs)

    if verbose:
        print("Loaded DataFrame shape: ", df.shape)

    return df


#===============================================================================



# TODO: Make split JSONL and split YAML functions behave as generic utils.
#===============================================================================
# 4| Split JSONL to DataFrames
#===============================================================================
def _split_JSONL_to_dfs(filepath: str | Path,
                        verbose: bool = False
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a JSONL file and split into metadata and trials DataFrames.
    Each JSONL record produces one metadata row and multiple trial rows.
    ------------------------------------
    Parameters:
        filepath: str or Path
            Path to the JSONL file.
        verbose: bool, optional
            If True, prints detailed output during loading. Default is False.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            A tuple containing the metadata DataFrame and the trials DataFrame.
    """
    if not isinstance(filepath, (str, Path)):
        raise ValueError("filepath must be a string or Path object.")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The specified JSONL file does not exist: {filepath}")

    with open(filepath) as f:
        records = [json.loads(line) for line in f]

    # Metadata: flatten each record's metadata into one row per record
    meta_df = pd.json_normalize([r["value"]["metadata"] for r in records])

    # Trials: collect all trials across all records, tag each with its timestamp
    all_trials = []
    for r in records:
        for trial in r["value"]["trials"]:
            trial["seconds"] = r.get("seconds")
            all_trials.append(trial)
    trials_df = pd.json_normalize(all_trials)

    if verbose:
        print(f"Loaded {len(records)} records")
        print(f"Metadata: {meta_df.shape}, Trials: {trials_df.shape}")

    return meta_df, trials_df




#===============================================================================




#===============================================================================
# 5| Split YAML to DataFrames
#===============================================================================
def _split_YAML_to_dfs(filepath: str | Path,
                       verbose: bool = False
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a YAML file and split into metadata and trials DataFrames.
    Each YAML record produces one metadata row and multiple trial rows.
    """
    with open(filepath, "r") as f:
        record = yaml.safe_load(f)

    # Metadata
    meta_df = pd.json_normalize(record["metadata"])
    
    # Trials
    trials_list = []
    for trial in record["trials"]:
        flat_trial = pd.json_normalize(trial).iloc[0]
        trials_list.append(flat_trial)
    trials_df = pd.DataFrame(trials_list)

    if verbose:
        print(f"Metadata shape: {meta_df.shape}, Trials shape: {trials_df.shape}")

    return meta_df, trials_df

#===============================================================================


collect_device_folders


#===============================================================================
# 6| Collect Device Folders
#===============================================================================
def _collect_device_folders(experiment_directory_path: str | Path,
                            device_type: str,
                            verbose: bool = False
                            ) -> list[Path]:
    '''
    Wrapper for collect_device_folders from harp_extender_core to maintain consistent naming and usage within this file. 
    Collects subfolders of a specified device type from an experimental directory.
    -----------------------------------
    Parameters:
        experiment_directory_path (str or Path): Path to the experiment directory.
        device_type (str): The type of device to filter folders by (e.g., 'Behavior').

    Returns:
        sorted([f for f in folder_path.glob(f'{device_type}*') if f.is_dir()], key=lambda p: p.name): A list of folder names that match the specified device type.

    '''
    if verbose: print(f"Collecting folders for device type '{device_type}' in '{experiment_directory_path}'")
    return collect_device_folders(experiment_directory_path= experiment_directory_path,
                                  device_type= device_type
                                  )


#===============================================================================





########################################################################################################################








########################################################################################################################
# Public Functions
########################################################################################################################




################################################################################
# Filetype Data Base Class
################################################################################
class FileTypeData:
    '''
    Class to manage the loading and processing of assorted data filetypes from HARP devices, such as CSV, JSONL, YAML, etc.
    '''

    def __init__(self,
                 experiment_directory_path: str | Path = None,      # Path to experimental directory (str or Path)
                 device_type: str = None,                           # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
                 file_type: str = None,                             # File type to load (str). E.g. 'csv', 'json', etc.
                 filepath: str | Path = None,                       # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
                 custom_kwargs: dict | None = None,                 # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If
                 rename_columns_dict: dict | None = None,           # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int.
                 rename_index_dict: dict | None = None,             # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
                 subfolder_index: int | None = None,                # Subfolder index to load (int). If folder_path contains multiple subfolders, specify which one to use (int). Default is first subfolder (0).
                 verbose: bool = False,                             # If True, prints detailed output during processing (bool)
                 ):
        
        super(FileTypeData, self).__init__()

        self.experiment_directory_path = experiment_directory_path
        self.device_type               = device_type
        self.file_type                 = file_type
        self.filepath                  = filepath
        self.rename_columns_dict       = rename_columns_dict
        self.rename_index_dict         = rename_index_dict
        self.verbose                   = verbose
        self.subfolder_index           = subfolder_index if subfolder_index is not None else 0

        # Set default custom kwargs based on file type

        if self.file_type == 'csv':
            self.custom_kwargs = custom_kwargs if custom_kwargs is not None else {
                'header': 0,
                'dtype': str,
                'engine': 'python',
                'sep': r',(?![^{]*})',  # Split on commas not inside braces
                'index_col': 0,
            }
        
        elif self.file_type == 'json':
            self.custom_kwargs = custom_kwargs if custom_kwargs is not None else {
                'lines': True,
                'dtype': str,
                'orient': 'records',
                'convert_dates': False,
                'precise_float': True,
            }
        
        elif self.file_type == 'jsonl':
            self.custom_kwargs = custom_kwargs if custom_kwargs is not None else {
                'lines': True,
                'dtype': str,
                'orient': 'records',
                'convert_dates': False,
                'precise_float': True,
            }
            if self.verbose:
                print(f'''This custom jsonl method does not currently support custom_kwargs, as it exists as a
                    specific use-case method. Please add and modify an appropriate method directly if needed.''')
            
        elif self.file_type in ['yml', 'yaml']:
            self.custom_kwargs = custom_kwargs if custom_kwargs is not None else {}
            if self.verbose:
                print(f'''This custom yaml method does not currently support custom_kwargs, as it exists as a
                    specific use-case method. Please add and modify an appropriate method directly if needed.''')
        else:
            raise ValueError(f'''Unsupported file_type: {self.file_type}. Supported types are 'csv', 'json', 
                             'jsonl', 'yml', and 'yaml'. Custom loading functions can be added as needed.''')
        
        self.df = self._get_df_from_filetype()

    ##===== Load Data into DataFrame ==============================
    def _get_df_from_filetype(self, 
                                verbose: bool | None = None
                                ) -> pd.DataFrame:
        '''
        Load data from a file into a pandas DataFrame based on the specified file type.
        -----------------------------------
        Returns:
            pd.DataFrame | dict[str, pd.DataFrame]: DataFrame containing the loaded data. 
            If file_type is 'jsonl' or 'yml'/'yaml', returns a dict with 'metadata' and 'trials' DataFrames.
        '''
        if verbose is None: verbose = self.verbose

        #==== Resolve filepath if not provided directly
        if self.filepath is None:
            if self.experiment_directory_path is not None and self.device_type is not None:
                subfolder_path = collect_device_folders(experiment_directory_path=self.experiment_directory_path,
                                                            device_type=self.device_type,
                                                            )
                if subfolder_path is None or len(subfolder_path) == 0:
                    raise ValueError(f"No subfolder found for device_type '{self.device_type}' in '{self.experiment_directory_path}'")
                if len(subfolder_path) > 1:
                    if verbose: print(f"Multiple subfolders found for device_type '{self.device_type}' in '{self.experiment_directory_path}': {subfolder_path}. Using subfolder index {self.subfolder_index}.")

                self.filepath = _collect_filetype_path(folder_path = subfolder_path[self.subfolder_index],
                                                        file_type = self.file_type,
                                                        return_type = 'first',
                                                        verbose = verbose   
                                                    )                                        
                if self.filepath is None:
                    raise ValueError(f"No {self.file_type} file found in subfolder '{subfolder_path[self.subfolder_index]}'")
                if verbose:
                    print(f"Found {self.file_type} file: {self.filepath}")
        
        #==== Load DataFrame based on file type
        if self.filepath is not None:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f'The specified {self.file_type} file does not exist: {self.filepath}')

            if self.file_type == 'csv':
                df = _CSV_to_df(filepath=self.filepath, csv_kwargs=self.custom_kwargs, verbose=verbose)
            
            elif self.file_type == 'json':
                df = _JSONL_to_df(filepath=self.filepath, jsonl_kwargs=self.custom_kwargs, verbose=verbose)
            
            elif self.file_type == 'jsonl':
                meta_df, trials_df = _split_JSONL_to_dfs(filepath=self.filepath, verbose=verbose)
                df = {'metadata': meta_df, 'trials': trials_df}
                if verbose:
                    print(f"Metadata DataFrame shape: {meta_df.shape}")
                    print(f"Trials DataFrame shape: {trials_df.shape}")
            
            elif self.file_type in ['yml', 'yaml']:
                meta_df, trials_df = _split_YAML_to_dfs(filepath=self.filepath, verbose=verbose)
                df = {'metadata': meta_df, 'trials': trials_df}
                if verbose:
                    print(f"Metadata DataFrame shape: {meta_df.shape}")
                    print(f"Trials DataFrame shape: {trials_df.shape}")
            else:
                raise ValueError(f'''Unsupported file_type: {self.file_type}. Supported types are 'csv', 'json', 
                                    'jsonl', 'yml', and 'yaml'. Custom loading functions can be added as needed.''')
           
            #==== Rename columns and index if specified
            if self.rename_columns_dict is not None:
                if isinstance(df, dict):
                    df = {k: v.rename(columns=self.rename_columns_dict) for k, v in df.items()}
                else:
                    df = df.rename(columns=self.rename_columns_dict)
                if verbose:
                    print(f"Renamed columns: {self.rename_columns_dict}")

            if self.rename_index_dict is not None:
                if isinstance(df, dict):
                    for v in df.values():
                        v.index.rename(self.rename_index_dict, inplace=True)
                else:
                    df.index.rename(self.rename_index_dict, inplace=True)
                if verbose:
                    print(f"Renamed index: {self.rename_index_dict}")
            
            return df


################################################################################




################################################################################
# Experiment Events
################################################################################
class ExperimentEvents(FileTypeData):
    '''
    Class to manage loading and processing of ExperimentEvents JSONL files from HARP devices.
    Inherits from filetype_data class.
    '''

    def __init__(self,
                 experiment_directory_path: str | Path = None,      # Path to experimental directory (str or Path)
                 device_type = 'ExperimentEvents',                           # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
                 file_type = 'csv',                             # File type to load (str). E.g. 'csv', 'json', etc.
                 filepath: str | Path = None,                       # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
                 custom_kwargs: dict | None = None,                 # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If
                 rename_columns_dict = {'Value': 'Event'},           # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int.
                 rename_index_dict = 'Time',             # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
                 subfolder_index: int | None = None,                # Subfolder index to load (int). If folder_path contains multiple subfolders, specify which one to use (int). Default is first subfolder (0).
                 verbose: bool = False,                             # If True, prints detailed output during processing (bool)                     # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
    ):
        super(ExperimentEvents, self).__init__(
            experiment_directory_path=experiment_directory_path,
            device_type=device_type,
            file_type=file_type,
            filepath=filepath,
            custom_kwargs=custom_kwargs,
            rename_columns_dict=rename_columns_dict,
            rename_index_dict=rename_index_dict,
            subfolder_index=subfolder_index,
            verbose=verbose)
        
        
        if self.file_type != 'csv':  # Updated to 'csv'
            raise ValueError(f"ExperimentEvents class only supports 'csv' file type. Provided: {self.file_type}")

        # Drop any unnecessary columns if present
        if 'Unamed: 2' in self.df.columns:
            self.df = self.df.drop(columns=['Unamed: 2'])
    


        if self.df is None:
            raise ValueError("Failed to load ExperimentEvents data.")

        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("ExperimentEvents data must be a pandas DataFrame.")
        if 'Event' not in self.df.columns:
            raise ValueError("ExperimentEvents DataFrame must contain 'Event' column.")
        if self.df.index.name != 'Time':
            raise ValueError("ExperimentEvents DataFrame index must be named 'Time'.")
        if self.verbose:
            print("ExperimentEvents data loaded successfully.")
            print("ExperimentEvents full DataFrame shape: ", self.df.shape)
        


################################################################################




################################################################################
# RotationData
################################################################################
class RotationData(FileTypeData):
    '''
    Class to manage loading and processing of RotationData CSV files from HARP devices.
    Inherits from filetype_data class.

    Device type must be one of InnerRotation, OuterRotation, or NosepokeRotation
    '''

    def __init__(self,
                experiment_directory_path: str | Path = None,      # Path to experimental directory (str or Path)
                device_type: str | None = None,                    # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
                file_type = 'csv',                                 # File type to load (str). E.g. 'csv', 'json', etc.
                filepath: str | Path = None,                       # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
                custom_kwargs: dict | None = None,                 # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If
                rename_columns_dict = {'Value': 'Rotation'},       # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int.
                rename_index_dict = 'Time',                        # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
                subfolder_index: int | None = None,                # Subfolder index to load (int). If folder_path contains multiple subfolders, specify which one to use (int). Default is first subfolder (0).
                angular_unit_conversion= None,                     # If provided, will convert rotation values to specified unit (str). Can be: 'deg2rad', 'rad2deg'.
                range= None,                                       # If provided, will wrap angular data within specified range. E.g. [0, 360], [-180, 180], [[0, 2*pi], [-pi, pi]] (list of two floats). If None, no wrapping is applied. Default is None.
                verbose: bool = False,                             # If True, prints detailed output during processing (bool)                     # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
                ):
        
        super(RotationData, self).__init__(
            experiment_directory_path=experiment_directory_path,
            device_type=device_type,
            file_type=file_type,
            filepath=filepath,
            custom_kwargs=custom_kwargs,
            rename_columns_dict=rename_columns_dict,
            rename_index_dict=rename_index_dict,
            subfolder_index=subfolder_index,
            verbose=verbose
        )

        self.angular_unit_conversion = angular_unit_conversion
        self.range = range

        if self.file_type != 'csv':
            raise ValueError(f"RotationData class only supports 'csv' file type. Provided: {self.file_type}")
        
        if self.device_type is not None:
            if self.device_type not in ['InnerRotation', 'OuterRotation', 'NosepokeRotation']:
                raise ValueError(f"RotationData device_type must be 'InnerRotation', 'OuterRotation', or 'NosepokeRotation'. Provided: {self.device_type}")
        
        if self.df is None:
            raise ValueError("Failed to load RotationData data.")
        
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("RotationData data must be a pandas DataFrame.")

        if 'Rotation' not in self.df.columns:
            raise ValueError("RotationData DataFrame must contain 'Rotation' column.")

        if self.df.index.name != 'Time':
            raise ValueError("RotationData DataFrame index must be named 'Time'.")

        #= Convert angular units if specified
        if self.angular_unit_conversion is not None:
            # Ensure Rotation column is numeric before applying conversions
            self.df['Rotation'] = pd.to_numeric(self.df['Rotation'], errors='coerce')
            if self.df['Rotation'].isna().any():
                bad_rows = int(self.df['Rotation'].isna().sum())
                raise ValueError(f"RotationData: 'Rotation' column contains {bad_rows} non-numeric value(s); cannot apply angular unit conversion.")
            if self.angular_unit_conversion == 'deg2rad':
                self.df['Rotation'] = np.deg2rad(self.df['Rotation'])
            elif self.angular_unit_conversion == 'rad2deg':
                self.df['Rotation'] = np.rad2deg(self.df['Rotation'])
            else:
                raise ValueError(f"Invalid angular_unit_conversion: {self.angular_unit_conversion}. Must be 'deg2rad' or 'rad2deg'. Provided: {self.angular_unit_conversion}")
            if self.verbose:
                print(f"Converted Rotation values using {self.angular_unit_conversion}")

        #= Wrap angular data within specified range if provided
        if self.range is not None:
            if isinstance(self.range, list) and len(self.range) == 2:
                min_val, max_val = self.range
                self.df['Rotation'] = ((self.df['Rotation'] - min_val) % (max_val - min_val)) + min_val
                if self.verbose:
                    print(f"Wrapped Rotation values within range {self.range}")
                    print(f'Min Angle after wrapping: {self.df["Rotation"].min()}')
                    print(f'Max Angle after wrapping: {self.df["Rotation"].max()}')
            else:
                raise ValueError(f"Invalid range: {self.range}. Must be a list of two floats. Provided: {self.range}")
            


################################################################################




################################################################################
# Video Data 
################################################################################
class VideoData(FileTypeData):
    '''
    Class to manage loading and processing of VideoData files from HARP devices.
    Inherits from FileTypeData class.
    '''

    def __init__(self,
                experiment_directory_path: str | Path = None,      # Path to experimental directory (str or Path)
                device_type = 'VideoData',                         # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
                file_type: str = None,                             # File type to load (str). E.g. 'csv', 'json', etc.
                filepath: str | Path = None,                       # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
                custom_kwargs: dict | None = None,                 # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If
                rename_columns_dict =  {'Value.ChunkData.FrameID': 'FrameID', 'Value.ChunkData.Timestamp': 'Timestamp'},                 # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int. 
                rename_index_dict = 'Time',                        # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
                subfolder_index: int | None = None,                # Subfolder index to load (int). If folder_path contains multiple subfolders, specify which one to use (int). Default is first subfolder (0).
                verbose: bool = False,                             # If True, prints detailed output during processing (bool)
                ):

        super(VideoData, self).__init__(
            experiment_directory_path=experiment_directory_path,
            device_type=device_type,
            file_type=file_type,
            filepath=filepath,
            custom_kwargs=custom_kwargs,
            rename_columns_dict=rename_columns_dict,
            rename_index_dict=rename_index_dict,
            subfolder_index=subfolder_index,
            verbose=verbose,
        )
        if self.file_type != 'csv':
            raise ValueError(f"VideoData class only supports 'csv' file type. Provided: {self.file_type}")

        if self.df is None:
            raise ValueError("Failed to load VideoData data.")

        if not isinstance(self.df, pd.DataFrame):
            raise ValueError(f"Invalid data format: {type(self.df)}. Expected: pd.DataFrame")
        if self.df.index.name != 'Time':
            raise ValueError("VideoData DataFrame index must be named 'Time'.")

        if self.df.empty:
            raise ValueError("VideoData DataFrame is empty.")
        


        if self.verbose:
            print("VideoData data loaded successfully.")
            print("VideoData full DataFrame shape: ", self.df.shape)

################################################################################




################################################################################
# Visual Environment
################################################################################
class VisualEnvironment(FileTypeData):
    '''
    Class to manage loading and processing of VideoData CSV files from HARP devices.
    Inherits from FileTypeData class.
    '''
    
    def __init__(self,
                experiment_directory_path: str | Path = None,      # Path to experimental directory (str or Path)
                device_type = 'VisualEnvironment',                         # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
                file_type: str = None,                             # File type to load (str). E.g. 'csv', 'json', etc.
                filepath: str | Path = None,                       # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
                custom_kwargs = {'names': ['Time', "Value", "Landmark", "Landmark_Proximal", "Gratings", "Firefly"],  # Add more names if needed
                                              'skiprows': 1,
                                              'header': None,
                                              'engine': "python",
                                              'index_col': 0  # Set the first column as the index
                                              },                      # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If None, will use default kwargs for file type.
                rename_columns_dict: dict | None = None,                 # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int. 
                rename_index_dict = 'Time',                        # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
                subfolder_index: int | None = None,                # Subfolder index to load (int). If folder_path contains multiple subfolders, specify which one to use (int). Default is first subfolder (0).
                verbose: bool = False,                             # If True, prints detailed output during processing (bool)
                ):

        super(VisualEnvironment, self).__init__(
            experiment_directory_path=experiment_directory_path,
            device_type=device_type,
            file_type=file_type,
            filepath=filepath,
            custom_kwargs=custom_kwargs,
            rename_columns_dict=rename_columns_dict,
            rename_index_dict=rename_index_dict,
            subfolder_index=subfolder_index,
            verbose=verbose,
        )
        if self.file_type != 'csv':
            raise ValueError(f"VisualEnvironment class only supports 'csv' file type. Provided: {self.file_type}")

        if self.df is None:
            raise ValueError("Failed to load VisualEnvironment data.")

        if not isinstance(self.df, pd.DataFrame):
            raise ValueError(f"Invalid data format: {type(self.df)}. Expected: pd.DataFrame")
        if self.df.index.name != 'Time':
            raise ValueError("VisualEnvironment DataFrame index must be named 'Time'.")

        if self.df.empty:
            raise ValueError("VisualEnvironment DataFrame is empty.")


        if self.verbose:
            print("VisualEnvironment data loaded successfully.")
            print("VisualEnvironment full DataFrame shape: ", self.df.shape)


################################################################################




################################################################################
# RingDebugData
################################################################################
class RingDebugData(FileTypeData):
    '''
    Class to manage loading and processing of RingDebugData CSV files from HARP devices.
    Inherits from filetype_data class.
    '''
    
    def __init__(self,
                 experiment_directory_path: str | Path = None,      # Path to experimental directory (str or Path)
                 device_type: str = 'ring-debug',                           # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
                 file_type: str = 'yml',                             # File type to load (str). E.g. 'csv', 'json', etc.
                 filepath: str | Path = None,                       # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
                 custom_kwargs: dict | None = None,                 # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If
                 rename_columns_dict: dict | None = None,           # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int.
                 rename_index_dict: dict | None = 'Time',             # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
                 subfolder_index: int | None = None,                # Subfolder index to load (int). If folder_path contains multiple subfolders, specify which one to use (int). Default is first subfolder (0).
                 verbose: bool = False,                             # If True, prints detailed output during processing (bool)
                 ):
                
        if filepath is None:
            try:
                filepath= f'{experimental_directory_path}/ring-debug.yml'
            except:
                raise ValueError(f'ring-debug.yml file not found in provided experimental_directory_path: {experimental_directory_path}. Ring-debug.yml is unique and does not typically reside in subfolder, which causes errors with the standard filetype_data class methods. Please provide full filepath if needed.')
        
        self.filepath= filepath
        
        super(RingDebugData, self).__init__(exoperiment_directory_path=experiment_directory_path,
                                            device_type=device_type,
                                            file_type=file_type,
                                            filepath=filepath,
                                            custom_kwargs=custom_kwargs,
                                            rename_columns_dict=rename_columns_dict,
                                            rename_index_dict=rename_index_dict,
                                            subfolder_index=subfolder_index,
                                            verbose=verbose)
        
        if self.file_type != 'yml':
            raise ValueError(f"RingDebugData class only supports 'yml' file type. Provided: {self.file_type}")

        if self.df is None:
            raise ValueError("Failed to load RingDebugData data.")

        if not isinstance(self.df, dict):
            raise ValueError(f"Invalid data format: {type(self.df)}. Expected: dict")

        if 'metadata' not in self.df or 'trials' not in self.df:
            raise ValueError("RingDebugData dict must contain 'metadata' and 'trials' keys.")

        if self.df['metadata'].empty:
            raise ValueError("RingDebugData 'metadata' DataFrame is empty.")

        if self.df['trials'].empty:
            raise ValueError("RingDebugData 'trials' DataFrame is empty.")

        if self.verbose:
            print("RingDebugData loaded successfully.")
            print("RingDebugData 'metadata' DataFrame shape: ", self.df['metadata'].shape)
            print("RingDebugData 'trials' DataFrame shape: ", self.df['trials'].shape)


################################################################################




########################################################################################################################






########################################################################################################################
# Wrappers for renamed, moved, or deprecated functions
########################################################################################################################




################################################################################
# Get Filetype Path from Folder
################################################################################


##--------------------------------------------------------------------------------   Original
# def get_filetype_path_from_folder(subfolder_path=None,    # Path to experimental directory (str or Path)
#                                    file_type= None,                     # File type to search for (str). E.g., 'csv', 'HDF5', 'YAML', 'JSON', 'TXT', etc.
#                                    return_type= 'first',                # Return type (str). Options: 'first', 'all', 'count'.
#                                    verbose= False,                       # If True, prints detailed output during search.
#                                    subfolder_n= 0                       # If subfolder_path contains multiple subfolders, specify which one to use (int). Default is 0 (first subfolder).
#                                    ):
#     '''
#     Get file paths of a specific type from an experimental directory.
#     -----------------------------------
#     Parameters:
#         subfolder_path (str or Path): Path to the subfolder within the experimental directory.
#         file_type (str): File type to search for (e.g., 'csv', 'HDF5', 'YAML', 'JSON', 'TXT').
#         return_type (str): Type of return value. Options are:
#             - 'first': Return the first matching file path or None if no files found.
#             - 'all': Return a list of all matching file paths.
#             - 'count': Return the count of matching files.
#         verbose (bool): If True, prints detailed output during the search process.
#         subfolder_n (int): If subfolder_path contains multiple subfolders, specify which one to use (int). Default is 0 (first subfolder).
#     Returns:
#         Depending on return_type:
#             - 'first': str or Path or None
#             - 'all': list of str or Path
#             - 'count': int
#     '''

#     if not subfolder_path:
#         raise ValueError("Subfolder path must be provided.")

#     if verbose:
#         print(f"Searching for {file_type} files in {subfolder_path}")

#     # Get all files of the specified type
#     file_paths = list(Path(subfolder_path).rglob(f"*.{file_type}")) if file_type else []

#     if return_type == 'first':
#         return file_paths[subfolder_n] if file_paths else None
#     elif return_type == 'all':
#         return file_paths
#     elif return_type == 'count':
#         return len(file_paths)
#     else:
#         raise ValueError(f"Invalid return_type: {return_type}. Must be 'first', 'all', or 'count'.")
################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original
# def CSV_data_to_df(filepath,
#                    csv_kwargs= None,
#                    verbose= False
#                    ):
#     '''
#     Load CSV data from a file into a pandas DataFrame.
#     -----------------------------------
#     Parameters:
#         filepath (str or Path): Path to the CSV file.
#         csv_kwargs (dict or None): Optional dictionary of keyword arguments to pass to pd.read_csv. If None, default arguments will be used.
#         verbose (bool): If True, prints detailed output during loading.
#     Returns:
#         pd.DataFrame: DataFrame containing the loaded CSV data.
#     '''
#     if not isinstance(filepath, (str, Path)):
#         raise ValueError("filepath must be a string or Path object.")
#     if not os.path.exists(filepath):
#         raise FileNotFoundError(f"The specified CSV file does not exist: {filepath}")
#     if csv_kwargs is None:
#         csv_kwargs= {'header': 0,
#                      'dtype': str,
#                      'engine': 'python',
#                      'sep': r',(?![^{]*})',  # Split on commas not inside braces
#                      'index_col': 0,
#                      }
#         if verbose:
#             print("Using default csv_kwargs: ", csv_kwargs)
#     else:
#         if verbose:
#             print("Using provided csv_kwargs: ", csv_kwargs)
#     df= pd.read_csv(filepath, **csv_kwargs)

#     if verbose:
#         print("Loaded DataFrame shape: ", df.shape)

#     return df
################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original
# def JSONL_data_to_df(filepath,
#                      jsonl_kwargs= None,    
#                      verbose= False
#                         ):
#     '''
#     Load JSONL data from a file into a pandas DataFrame.
#     -----------------------------------
#     Parameters:
#         filepath (str or Path): Path to the JSONL file.
#         jsonl_kwargs (dict or None): Optional dictionary of keyword arguments to pass to pd.read_json. If None, default arguments will be used.
#         verbose (bool): If True, prints detailed output during loading.
#     Returns:
#         pd.DataFrame: DataFrame containing the loaded JSONL data.
#     '''
#     if not isinstance(filepath, (str, Path)):
#         raise ValueError("filepath must be a string or Path object.")
#     if not os.path.exists(filepath):
#         raise FileNotFoundError(f"The specified JSONL file does not exist: {filepath}")
#     if jsonl_kwargs is None:
#         jsonl_kwargs= {'lines': True,
#                        'dtype': str,
#                        'orient': 'records',
#                        'convert_dates': False,
#                        'precise_float': True,
#                        }
#         if verbose:
#             print("Using default jsonl_kwargs: ", jsonl_kwargs)
#     else:
#         if verbose:
#             print("Using provided jsonl_kwargs: ", jsonl_kwargs)
#     df= pd.read_json(filepath, **jsonl_kwargs)

#     if verbose:
#         print("Loaded DataFrame shape: ", df.shape)

#     return df
################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original

# def split_jsonl_to_dfs(filepath, verbose=False):
#     """
#     Load a JSONL file and split into metadata and trials DataFrames.
#     Each JSONL record produces one metadata row and multiple trial rows.
#     """
#     with open(filepath) as f:
#         records = [json.loads(line) for line in f]

#     # Base DataFrame for quick access
#     df = pd.json_normalize(records)

#     # Metadata: flatten the value.metadata dict
#     meta_df = pd.json_normalize([r["value"]["metadata"] for r in records])
#     meta_df.columns = [f"{c}" for c in meta_df.columns]

#     # Trials: explode into separate rows
#     trials_list = []
#     for r in records:
#         seconds = r.get("seconds")
#         for trial in r["value"]["trials"]:
#             flat_trial = pd.json_normalize(trial).iloc[0]
#             flat_trial["seconds"] = seconds
#             trials_list.append(flat_trial)

#     trials_df = pd.DataFrame(trials_list)

#     if verbose:
#         print(f"Loaded {len(records)} records")
#         print(f"Metadata shape: {meta_df.shape}, Trials shape: {trials_df.shape}")

#     return meta_df, trials_df

################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original
# def split_yaml_to_dfs(filepath, verbose=False):
#     """
#     Load a YAML file and split into metadata and trials DataFrames.
#     Each YAML record produces one metadata row and multiple trial rows.
#     """
#     with open(filepath, "r") as f:
#         record = yaml.safe_load(f)

#     # Metadata
#     meta_df = pd.json_normalize(record["metadata"])
    
#     # Trials
#     trials_list = []
#     for trial in record["trials"]:
#         flat_trial = pd.json_normalize(trial).iloc[0]
#         trials_list.append(flat_trial)
#     trials_df = pd.DataFrame(trials_list)

#     if verbose:
#         print(f"Metadata shape: {meta_df.shape}, Trials shape: {trials_df.shape}")

#     return meta_df, trials_df
################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original
# class filetype_data:
#     '''
#     Class to manage loading and processing of various file types from HARP devices.
#     '''

#     def __init__(self,
#                  experimental_directory_path= None,         # Path to experimental directory (str or Path)  
#                  device_type= None,                         # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
#                  file_type= None,                           # File type to load (str). E.g. 'csv', 'json', etc.
#                  filepath= None,                            # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
#                  custom_kwargs= None,                       # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If None, will use default kwargs for file type.
#                  rename_columns_dict= None,                 # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int. 
#                  rename_index_dict= None,                   # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
#                  verbose= False,                            # If True, prints detailed output during processing (bool)
#                  subfolder_n= None                          # Subfolder number to load (int or None). If None, will load first subfolder found with matching device_type prefix. Not currently used.
#                  ):
        
        
#         super(filetype_data, self).__init__()

#         self.experimental_directory_path= experimental_directory_path
#         self.device_type= device_type
#         self.file_type= file_type
#         self.filepath= filepath

#         self.verbose= verbose
#         self.rename_columns_dict= rename_columns_dict
#         self.rename_index_dict= rename_index_dict
#         # self.df= None                                       # Test Hotfix for Unbound Local Error

#         if file_type == 'csv':
#             if custom_kwargs is None:
#                 self.custom_kwargs= {'header': 0,
#                                   'dtype': str,
#                                   'engine': 'python',
#                                   'sep': r',(?![^{]*})',  # Split on commas not inside braces
#                                   'index_col': 0,
#                                   }
#             else:
#                 self.custom_kwargs= custom_kwargs
        
#         if file_type == 'json':
#             if custom_kwargs is None:
#                 self.custom_kwargs= {'lines': True,
#                                'dtype': str,
#                                'orient': 'records',
#                                'convert_dates': False,
#                                'precise_float': True,
#                                }
#             else:
#                 self.custom_kwargs= custom_kwargs
#         # if file_type == 'jsonl':
#         #     if custom_kwargs is None:
#         #         self.custom_kwargs= {'lines': True,
#         #                        'dtype': str,
#         #                        'orient': 'records',
#         #                        'convert_dates': False,
#         #                        'precise_float': True,
#         #                        }
#         #     else:
#         #         self.custom_kwargs= custom_kwargs

#         if file_type == 'jsonl':
#             if custom_kwargs is None:
#                 self.custom_kwargs= {'lines': True,
#                                'dtype': str,
#                                'orient': 'records',
#                                'convert_dates': False,
#                                'precise_float': True,
#                                }
#             else:
#                 self.custom_kwargs= custom_kwargs
#             if self.verbose:
#                 print('This custom jsonl method does not currently support custom_kwargs, as it exists as a specific use-case method. Please add and modify an appropriate method directly if needed.')
            
#         if file_type == 'yml' or file_type == 'yaml':
#             if custom_kwargs is None:
#                 self.custom_kwargs= {}
#             else:
#                 self.custom_kwargs= custom_kwargs
#             if self.verbose:
#                 print('This custom yaml method does not currently support custom_kwargs, as it exists as a specific use-case method. Please add and modify an appropriate method directly if needed.')


#         if file_type not in ['csv', 'json', 'jsonl', 'yml', 'yaml']:
#             raise ValueError(f"Unsupported file_type: {file_type}. Supported types are 'csv', 'json', and 'jsonl'. Custom loading functions can be added as needed.")



#         if subfolder_n is not None:
#             self.n= subfolder_n
#         else:
#             self.n= 0  # Default to first subfolder if not specified

#         self.df= self.get_df_from_filetype()
#     ##===== Load Data into DataFrame ==============================
#     def get_df_from_filetype(self):
#         '''
#         Load data from a file into a pandas DataFrame based on the specified file type.
#         -----------------------------------
#         Parameters:
#             None
#         Returns:
#             pd.DataFrame: DataFrame containing the loaded data.
#         '''

#         if self.filepath is None:
#             if self.experimental_directory_path is not None and self.device_type is not None:
#                 subfolder_path=  get_device_subfolders(experiment_directory_path= self.experimental_directory_path,
#                                                          device_type= self.device_type)
#                 if subfolder_path is None:
#                     raise ValueError(f"No subfolder found for device_type '{self.device_type}' in '{self.experimental_directory_path}'")
#                 if len(subfolder_path) > 1:
#                     raise ValueError(f"Multiple subfolders found for device_type '{self.device_type}' in '{self.experimental_directory_path}': {subfolder_path}")
                
#                 self.filepath= get_filetype_path_from_folder(subfolder_path= subfolder_path[self.n],
#                                                             file_type= self.file_type,
#                                                             return_type= 'first',
#                                                             verbose= self.verbose,
#                                                             subfolder_n= self.n
#                                                               )
#                 if self.filepath is None:
#                     raise ValueError(f"No {self.file_type} file found in subfolder '{subfolder_path[self.n]}'")
#                 if self.verbose:
#                     print(f"Found {self.file_type} file: {self.filepath}")


#         if self.filepath is not None:
#             if not os.path.exists(self.filepath):
#                 raise FileNotFoundError(f"The specified {self.file_type} file does not exist: {self.filepath}")
#             if self.verbose:
#                 print(f"Loading {self.file_type} file: {self.filepath}")

#             if self.file_type == 'csv':
#                 df= CSV_data_to_df(filepath= self.filepath,
#                                    csv_kwargs= self.custom_kwargs,
#                                    verbose= self.verbose
#                                     )
#             if self.file_type == 'json':
#                 df= JSONL_data_to_df(filepath= self.filepath,
#                                    jsonl_kwargs= self.custom_kwargs,
#                                    verbose= self.verbose
#                                     )
#             # if self.file_type == 'jsonl':
#             #     df= self.JSONL_data_to_df_normalised(filepath= self.filepath,
#             #                        jsonl_kwargs= self.custom_kwargs,
#             #                        verbose= self.verbose
#             #                         )
            
#             if self.file_type == 'jsonl':
#                 meta_df, trials_df = split_jsonl_to_dfs(filepath= self.filepath,
#                                                         verbose= self.verbose
#                                                         )
#                 df= {'metadata': meta_df, 'trials': trials_df}
#                 if self.verbose:
#                     print("Metadata DataFrame shape: ", meta_df.shape)
#                     print("Trials DataFrame shape: ", trials_df.shape)
                    
#                 return df
            
#             if self.file_type == 'yml' or self.file_type == 'yaml':
#                 meta_df, trials_df = split_yaml_to_dfs(filepath= self.filepath,
#                                                         verbose= self.verbose
#                                                         )
#                 if self.verbose:
#                     print("Metadata DataFrame shape: ", meta_df.shape)
#                     print("Trials DataFrame shape: ", trials_df.shape)
#                     df= {'metadata': meta_df, 'trials': trials_df}
#                 return df

#             if self.file_type not in ['csv', 'json', 'jsonl', 'yml', 'yaml']:
#                 raise ValueError(f"Unsupported file_type: {self.file_type}. Supported types are 'csv' and 'json'. Custom loading functions can be added as needed.")
            
#             if self.rename_columns_dict is not None:
#                 df= df.rename(columns= self.rename_columns_dict)
#                 if self.verbose:
#                     print(f"Renamed columns: {self.rename_columns_dict}")

#             if self.rename_index_dict is not None:
#                 df.index.rename(self.rename_index_dict, inplace= True)
#                 #df= df.rename(index= self.rename_index_dict)
#                 if self.verbose:
#                     print(f"Renamed index: {self.rename_index_dict}")
#             return df

#     def JSONL_data_to_df_normalised(self, filepath, jsonl_kwargs=None, verbose=False, normalise=True):
#             """Load JSON/JSONL data and optionally normalize nested structures."""
#             jsonl_kwargs = jsonl_kwargs
#             if verbose:
#                 print(f"Loading JSON/JSONL with kwargs: {jsonl_kwargs}")

#             # Load raw JSONL or JSON
#             with open(filepath, 'r') as f:
#                 records = [json.loads(line) for line in f]

#             df = pd.json_normalize(records)

#             if normalise is True:
#                 # Explode trials if present
#                 if 'value.trials' in df.columns:
#                     df = df.explode('value.trials', ignore_index=True)
#                     trial_df = pd.json_normalize(df['value.trials'])
#                     df = pd.concat([df.drop(columns=['value.trials']), trial_df], axis=1)

#                 # Flatten metadata if present
#                 if 'value.metadata' in df.columns:
#                     metadata_df = pd.json_normalize(df['value.metadata'])
#                     df = pd.concat([df.drop(columns=['value.metadata']), metadata_df], axis=1)

#                 if verbose:
#                     print(f"Normalized JSON/JSONL DataFrame to shape: {df.shape}")

#             return df
################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original
# class ExperimentEvents(filetype_data):
#     '''
#     Class to manage loading and processing of ExperimentEvents JSONL files from HARP devices.
#     Inherits from filetype_data class.
#     '''

#     def __init__(self,
#                  experimental_directory_path= None,         # Path to experimental directory (str or Path)  
#                  device_type= 'ExperimentEvents',          # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
#                  filepath= None,                            # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
#                  custom_kwargs= None,                       # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If None, will use default kwargs for file type.
#                  rename_columns_dict= {'Value': 'Event'},                 # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int. 
#                  rename_index_dict= 'Time',                   # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
#                  verbose= False,                            # If True, prints detailed output during processing (bool)
#                  subfolder_n= None                          # Subfolder number to load (int or None). If None, will load first subfolder found with matching device_type prefix. Not currently used.
#                  ):
        
        
#         super(ExperimentEvents, self).__init__(experimental_directory_path= experimental_directory_path,
#                                                device_type= device_type,
#                                                file_type= 'csv',
#                                                filepath= filepath,
#                                                custom_kwargs= custom_kwargs,
#                                                rename_columns_dict= rename_columns_dict,
#                                                rename_index_dict= rename_index_dict,
#                                                verbose= verbose,
#                                                subfolder_n= subfolder_n
#                                                 )


#         if self.file_type != 'csv':  # Updated to 'csv'
#             raise ValueError(f"ExperimentEvents class only supports 'csv' file type. Provided: {self.file_type}")

#         # Drop any unnecessary columns if present
#         if 'Unamed: 2' in self.df.columns:
#             self.df = self.df.drop(columns=['Unamed: 2'])
    


#         if self.df is None:
#             raise ValueError("Failed to load ExperimentEvents data.")

#         if not isinstance(self.df, pd.DataFrame):
#             raise ValueError("ExperimentEvents data must be a pandas DataFrame.")
#         if 'Event' not in self.df.columns:
#             raise ValueError("ExperimentEvents DataFrame must contain 'Event' column.")
#         if self.df.index.name != 'Time':
#             raise ValueError("ExperimentEvents DataFrame index must be named 'Time'.")
#         if self.verbose:
#             print("ExperimentEvents data loaded successfully.")
#             print("ExperimentEvents full DataFrame shape: ", self.df.shape)
################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original
# class RotationData(filetype_data):
#     '''
#     Class to manage loading and processing of RotationData CSV files from HARP devices.
#     Inherits from filetype_data class.
#     '''

#     def __init__(self,
#                  experimental_directory_path= None,         # Path to experimental directory (str or Path)  
#                  device_type= None,                   # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods. Can be either InnerRotation, OuterRotation, or NosepokeRotation.
#                  filepath= None,                            # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
#                  custom_kwargs= None,                       # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If None, will use default kwargs for file type.
#                  rename_columns_dict= {'Value': 'Rotation'},# Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int. 
#                  rename_index_dict= 'Time',                 # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
#                  verbose= False,                            # If True, prints detailed output during processing (bool)
#                  subfolder_n= None,                          # Subfolder number to load (int or None). If None, will load first subfolder found with matching device_type prefix. Not currently used.
#                  angular_unit_conversion= None,             # If provided, will convert rotation values to specified unit (str). Can be: 'deg2rad', 'rad2deg'.
#                  range= None,                               # If provided, will wrap angular data within specified range. E.g. [0, 360], [-180, 180], [[0, 2*pi], [-pi, pi]] (list of two floats). If None, no wrapping is applied. Default is None.
#     ):


#         super(RotationData, self).__init__(experimental_directory_path= experimental_directory_path,
#                                            device_type= device_type,
#                                            file_type= 'csv',
#                                            filepath= filepath,
#                                            custom_kwargs= custom_kwargs,
#                                            rename_columns_dict= rename_columns_dict,
#                                            rename_index_dict= rename_index_dict,
#                                            verbose= verbose,
#                                            subfolder_n= subfolder_n
#                                             )
        
#         self.angular_unit_conversion= angular_unit_conversion
#         self.range= range


#         if self.file_type != 'csv':
#             raise ValueError(f"RotationData class only supports 'csv' file type. Provided: {self.file_type}")
        
#         if self.device_type is not None:
#             if self.device_type not in ['InnerRotation', 'OuterRotation', 'NosepokeRotation']:
#                 raise ValueError(f"RotationData device_type must be 'InnerRotation', 'OuterRotation', or 'NosepokeRotation'. Provided: {self.device_type}")
        
#         if self.df is None:
#             raise ValueError("Failed to load RotationData data.")
        
#         if not isinstance(self.df, pd.DataFrame):
#             raise ValueError("RotationData data must be a pandas DataFrame.")

#         if 'Rotation' not in self.df.columns:
#             raise ValueError("RotationData DataFrame must contain 'Rotation' column.")

#         if self.df.index.name != 'Time':
#             raise ValueError("RotationData DataFrame index must be named 'Time'.")

#         #= Convert angular units if specified
#         if self.angular_unit_conversion is not None:
#             # Ensure Rotation column is numeric before applying conversions
#             self.df['Rotation'] = pd.to_numeric(self.df['Rotation'], errors='coerce')
#             if self.df['Rotation'].isna().any():
#                 bad_rows = int(self.df['Rotation'].isna().sum())
#                 raise ValueError(f"RotationData: 'Rotation' column contains {bad_rows} non-numeric value(s); cannot apply angular unit conversion.")
#             if self.angular_unit_conversion == 'deg2rad':
#                 self.df['Rotation'] = np.deg2rad(self.df['Rotation'])
#             elif self.angular_unit_conversion == 'rad2deg':
#                 self.df['Rotation'] = np.rad2deg(self.df['Rotation'])
#             else:
#                 raise ValueError(f"Invalid angular_unit_conversion: {self.angular_unit_conversion}. Must be 'deg2rad' or 'rad2deg'. Provided: {self.angular_unit_conversion}")
#             if self.verbose:
#                 print(f"Converted Rotation values using {self.angular_unit_conversion}")

#         #= Wrap angular data within specified range if provided
#         if self.range is not None:
#             if isinstance(self.range, list) and len(self.range) == 2:
#                 min_val, max_val = self.range
#                 self.df['Rotation'] = ((self.df['Rotation'] - min_val) % (max_val - min_val)) + min_val
#                 if self.verbose:
#                     print(f"Wrapped Rotation values within range {self.range}")
#                     print(f'Min Angle after wrapping: {self.df["Rotation"].min()}')
#                     print(f'Max Angle after wrapping: {self.df["Rotation"].max()}')
#             else:
#                 raise ValueError(f"Invalid range: {self.range}. Must be a list of two floats. Provided: {self.range}")
            
################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original
# class VisualEnvironment(filetype_data):
#     '''
#     Class to manage loading and processing of VisualEnvironment csv files from HARP devices.
#     Inherits from filetype_data class.
#     '''

#     def __init__(self,
#                  experimental_directory_path= None,         # Path to experimental directory (str or Path)  
#                  device_type= 'VisualEnvironment',          # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
#                  filepath= None,                            # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
#                  custom_kwargs = {'names': ['Time', "Value", "Landmark", "Landmark_Proximal", "Gratings", "Firefly"],  # Add more names if needed
#                                               'skiprows': 1,
#                                               'header': None,
#                                               'engine': "python",
#                                               'index_col': 0  # Set the first column as the index
#                                               },                      # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If None, will use default kwargs for file type.
#                  rename_columns_dict= None,                 # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int. 
#                  rename_index_dict= None,                   # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
#                  verbose= False,                            # If True, prints detailed output during processing (bool)
#                  subfolder_n= None                          # Subfolder number to load (int or None). If None, will load first subfolder found with matching device_type prefix. Not currently used.
#                  ):
        
        
#         super(VisualEnvironment, self).__init__(experimental_directory_path= experimental_directory_path,
#                                                device_type= device_type,
#                                                file_type= 'csv',
#                                                filepath= filepath,
#                                                custom_kwargs= custom_kwargs,
#                                                rename_columns_dict= rename_columns_dict,
#                                                rename_index_dict= rename_index_dict,
#                                                verbose= verbose,
#                                                subfolder_n= subfolder_n
#                                                 )


#         if self.file_type != 'csv':
#             raise ValueError(f"VisualEnvironment class only supports 'csv' file type. Provided: {self.file_type}")

#         if self.df is None:
#             raise ValueError("Failed to load VisualEnvironment data.")

#         if not isinstance(self.df, pd.DataFrame):
#             raise ValueError(f"Invalid data format: {type(self.df)}. Expected: pd.DataFrame")
#         if self.df.index.name != 'Time':
#             raise ValueError("VisualEnvironment DataFrame index must be named 'Time'.")

#         if self.df.empty:
#             raise ValueError("VisualEnvironment DataFrame is empty.")


#         if self.verbose:
#             print("VisualEnvironment data loaded successfully.")
#             print("VisualEnvironment full DataFrame shape: ", self.df.shape)


################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original

# class VideoData(filetype_data):
#     '''
#     Class to manage loading and processing of VideoData CSV files from HARP devices.
#     Inherits from filetype_data class.
#     '''

#     def __init__(self,
#                  experimental_directory_path= None,         # Path to experimental directory (str or Path)  
#                  device_type= 'VideoData',                      # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
#                  filepath= None,                            # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
#                  custom_kwargs= None,                       # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If None, will use default kwargs for file type.
#                  rename_columns_dict= {'Value.ChunkData.FrameID': 'FrameID', 'Value.ChunkData.Timestamp': 'Timestamp'},                 # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int. 
#                  rename_index_dict= 'Time',                 # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
#                  verbose= False,                            # If True, prints detailed output during processing (bool)
#                  subfolder_n= None                          # Subfolder number to load (int or None). If None, will load first subfolder found with matching device_type prefix. Not currently used.
#                  ):
        
        
#         super(VideoData, self).__init__(experimental_directory_path= experimental_directory_path,
#                                            device_type= device_type,
#                                            file_type= 'csv',
#                                            filepath= filepath,
#                                            custom_kwargs= custom_kwargs,
#                                            rename_columns_dict= rename_columns_dict,
#                                            rename_index_dict= rename_index_dict,
#                                            verbose= verbose,
#                                            subfolder_n= subfolder_n
#                                             )


#         if self.file_type != 'csv':
#             raise ValueError(f"VideoData class only supports 'csv' file type. Provided: {self.file_type}")

#         if self.df is None:
#             raise ValueError("Failed to load VideoData data.")

#         if not isinstance(self.df, pd.DataFrame):
#             raise ValueError(f"Invalid data format: {type(self.df)}. Expected: pd.DataFrame")
#         if self.df.index.name != 'Time':
#             raise ValueError("VideoData DataFrame index must be named 'Time'.")

#         if self.df.empty:
#             raise ValueError("VideoData DataFrame is empty.")
        


#         if self.verbose:
#             print("VideoData data loaded successfully.")
#             print("VideoData full DataFrame shape: ", self.df.shape)
################################################################################




################################################################################
#
################################################################################


##--------------------------------------------------------------------------------   Original

# class RingDebugData(filetype_data):
#     '''
#     Class to manage loading and processing of RingDebugData CSV files from HARP devices.
#     Inherits from filetype_data class.
#     '''

#     def __init__(self,
#                  experimental_directory_path= None,         # Path to experimental directory (str or Path)  
#                  device_type= 'ring-debug',                  # Subfolder Prefix type to load (str). Must be exact prefix. E.g. 'Behavior' will load 'Behavior0', 'Behavior1', etc. Should not be device, but device_type used for consistency across methods.
#                  filepath= None,                            # Path to file (str or Path). Used in place of experimental_directory_path and device_type if provided.
#                  custom_kwargs= None,                       # Dict of custom kwargs to pass to file loading function (dict). E.g. {'delimiter': ',', 'header': 0,...}. If None, will use default kwargs for file type.
#                  rename_columns_dict= None,                 # Dict of columns to rename (dict of str: str). E.g. {'OldName': 'NewName',...}. Default columns should be Time and Value, with Value being the main data column and Time being the index column with values in seconds as float or int. 
#                  rename_index_dict= 'Time',                 # Dict of index values to rename (dict of str: str). E.g. {'OldIndexValue': 'NewIndexValue',...}. Default index should be Time with values in seconds as float or int.
#                  verbose= False,                            # If True, prints detailed output during processing (bool)
#                  subfolder_n= None                          # Subfolder number to load (int or None). If None, will load first subfolder found with matching device_type prefix. Not currently used.
#                  ):
        
#         if filepath is None:
#             try:
#                 filepath= f'{experimental_directory_path}/ring-debug.yml'
#             except:
#                 raise ValueError(f'ring-debug.yml file not found in provided experimental_directory_path: {experimental_directory_path}. Ring-debug.yml is unique and does not typically reside in subfolder, which causes errors with the standard filetype_data class methods. Please provide full filepath if needed.')
        
#         self.filepath= filepath


#         super(RingDebugData, self).__init__(experimental_directory_path= experimental_directory_path,
#                                            device_type= device_type,
#                                            file_type= 'yml',
#                                            filepath= filepath,
#                                            custom_kwargs= custom_kwargs,
#                                            rename_columns_dict= rename_columns_dict,
#                                            rename_index_dict= rename_index_dict,
#                                            verbose= verbose,
#                                            subfolder_n= subfolder_n
#                                             )


#         if self.file_type != 'yml':
#             raise ValueError(f"RingDebugData class only supports 'yml' file type. Provided: {self.file_type}")

#         if self.df is None:
#             raise ValueError("Failed to load RingDebugData data.")

#         if not isinstance(self.df, dict):
#             raise ValueError(f"Invalid data format: {type(self.df)}. Expected: dict")

#         if 'metadata' not in self.df or 'trials' not in self.df:
#             raise ValueError("RingDebugData dict must contain 'metadata' and 'trials' keys.")

#         if self.df['metadata'].empty:
#             raise ValueError("RingDebugData 'metadata' DataFrame is empty.")

#         if self.df['trials'].empty:
#             raise ValueError("RingDebugData 'trials' DataFrame is empty.")

#         if self.verbose:
#             print("RingDebugData loaded successfully.")
#             print("RingDebugData 'metadata' DataFrame shape: ", self.df['metadata'].shape)
#             print("RingDebugData 'trials' DataFrame shape: ", self.df['trials'].shape)


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

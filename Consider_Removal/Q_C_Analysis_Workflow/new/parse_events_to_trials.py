'''
Parser for converting DataFrame containing event logs to a DataFrame with trial-level information.

The DataFrame will have the following columns:

1) trial_index:
    - The index of the trial within the session, starting at 1 for the first trial.

2) start_time:
    - The timestamp of the trial's start. 
    - For the first trial in a session, this is given by the log "Start trial logic".
    - For all subsequent trials, this is given by the end time of the previous trial, plus 1 second.

3) end_time:
    - The timestamp of the trial's end, given by the first occurrence of a "Poke:*" log after the trial's start time.

4) TTP:
    - Time to poke: the duration between the "Target zone triggered" log after the trial's start time and the first "Poke:*" log (end_time).

5) Outcome:
    - The outcome of the trial. This is determined by the tuple with shape "Poke: {Success=True, ChosenPort=16, CorrectPort=16-}". 
    - If Success = True, the outcome is "Success".
    - If Success = False, and Chosenport = -1 (i.e, no poke), the outcome is "Miss".
    - If Success = False, and Chosenport != -1, the outcome is "Failure".

6) LED:
    - If there is a "NosePokesLED ON" log between the trial's start time and end time, the LED value is "ON". Otherwise, it is "OFF".

7) Angle Offset:
    - The angle offset beteen the correct poke and the chosen poke. This should be calculated as +/- 180 degrees, depending on the direction of the offset.
    - The number nosepokes in the arena should be given as an input argument. As these are evenly spaced, the angle offset can be calculated as:
        angle_offset = (ChosenPort - CorrectPort) * (360 / number_of_poke_ports)
    - If there is no poke (ChosenPort = -1), the angle offset should be set to NaN.

8) Target Zone Size:
    - The size of the target zone for the trial, determined by the log "Target zone available (X Y : 740 687 - Radius : 150)".
    - The value is given by the radius value in the log (e.g., 150 in the example above). If there is no such log between the trial's start time and end time, the Target Zone Size is NaN.
    - Note: The target zone size can vary across trials, and is not necessarily the same for all trials within a session.

'''

import pandas as pd
import numpy as np
from pathlib import Path




#====================================================================================
# Helpers                                                                       
#====================================================================================

def _within_window(df: pd.DataFrame,
                   start_time: float,
                   end_time: float,
                   ) -> pd.DataFrame:
    '''
    Return rows where the index (timestamp) is between start_time and end_time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a numeric time index. 
    start_time : float
        The start time of the window.
    end_time : float
        The end time of the window.
    
    Returns
    -------
    pd.DataFrame
        A subset of the input DataFrame containing only rows where the index is between start_time and end_time.
    ''' 
    return df[(df.index >= start_time) & (df.index <= end_time)]



def _get_start_time(events_df: pd.DataFrame = None,
                    trial_index: int = None,
                    previous_trial_end_time: float | None = None,
                    ) -> float:
    '''
    Return the start time of a trial.

    Parameters
    ----------
    events_df : pd.DataFrame
        A DataFrame containing event logs for a session.
    trial_index : int
        The index of the trial within the session.
    previous_trial_end_time : float | None
        The end time of the previous trial. 
        This is used to calculate the start time of the current trial, as the end time of the previous trial plus 1 second. 
        For the first trial in a session, this should be None, and the start time will be determined by the log "Start trial logic".
    
    Returns
    -------
    float
        The start time of the trial.
    '''
    if trial_index == 0:                                    # noqa: SIM108 | This seems unnecessary 
        # For the first trial, get the start time from the "Start trial logic" log using pd.Series.str.contains to find 
        # the row containing the log, and return the index (timestamp) of that row.
        start_time = events_df[events_df['Event'].str.contains("Start trial logic", na = False)].index[0]
    else:
        # For subsequent trials, the start time is the end time of the previous trial plus 1 second.
        start_time = previous_trial_end_time + 1

    return start_time

def _get_outcome(poke_event: str
) -> str:
    '''
    Return the outcome of a trial based on the poke event string.

    Parameters
    ----------
    poke_event : str
        The event string containing the poke outcome information, e.g., "Poke: {Success=True, ChosenPort=16, CorrectPort=16-}".
        Outcomes are determined as follows:
        - If Success = True, the outcome is "Success".
        - If Success = False, and Chosenport = -1 (i.e, no poke), the outcome is "Miss".
        - If Success = False, and Chosenport != -1, the outcome is "Failure".
    
    Returns
    -------
    str
        The outcome of the trial: "Success", "Miss", or "Failure".
    '''

    if "Success=True" in poke_event:
        return "Success"
    elif "Success=False" in poke_event and "ChosenPort=-1" in poke_event:
        return "Miss"
    elif "Success=False" in poke_event and "ChosenPort=-1" not in poke_event:
        return "Failure"
    else:
        raise ValueError(f"Unexpected poke event format: {poke_event}")

def _get_TTP(window: pd.DataFrame,
             ) -> float:
    '''
    Return the time to poke (TTP) for a trial, calculated as the duration between the "Target zone triggered" log and 
    the first "Poke:*" log (end_time) within a given window.

    Parameters
    ----------
    window : pd.DataFrame
        A DataFrame containing event logs for a trial, filtered to the time window between the trial's start time and end time.
    
    Returns
    -------
    float
        The time to poke (TTP) for the trial, 
        calculated as the duration between the "Target zone triggered" log and the first "Poke:*" log (end_time) within the given window. 
        If either log is missing, return NaN.
    '''
    try:
        tz_trig_time = window[window['Event'].str.contains("Target zone triggered", na = False)].index[0]
        poke_time = window[window['Event'].str.contains("Poke:", na = False)].index[0]
        TTP = poke_time - tz_trig_time
        return TTP
    except IndexError:
        # If either log is missing, return NaN.
        return np.nan

def _get_LED(window: pd.DataFrame
) -> str:
    '''
    Return the LED status for a trial, determined by the presence of a "NosePokesLED ON" log within a given window.

    Parameters
    ----------
    window : pd.DataFrame
        A DataFrame containing event logs for a trial, filtered to the time window between the trial's start time and end time.
    
    Returns
    -------
    str
        The LED status for the trial: "ON" if there is a "NosePokesLED ON" log within the window, otherwise "OFF".
    '''
    if any(window['Event'].str.contains("NosePokesLED ON", na = False)):
        return "ON"
    else:
        return "OFF"

def _get_angle_offset(window: pd.DataFrame,
                      nosepoke_count: int,
                        ) -> float:
    '''
    Return the signed angle offset (+/- 180 degrees) for a trial, calculated as the angle between the correct poke and the chosen poke, 
    based on the "Poke: {Success=True, ChosenPort=16, CorrectPort=16-}" log within a given window.

    Parameters
    ----------
    window : pd.DataFrame
        A DataFrame containing event logs for a trial, filtered to the time window between the trial's start time and end time.
    nosepoke_count : int
        The number of nosepoke ports in the arena. This is used to calculate the angle offset, as the ports are evenly spaced around the arena.
    
    Returns
    -------
    float
        The angle offset for the trial, calculated as:
        angle_offset = (ChosenPort - CorrectPort) * (360 / nosepoke_count)
        If there is no poke (ChosenPort = -1), return NaN.
    '''
    # Get the poke event string from the window. This should be the first occurrence of a "Poke:" log after the trial's start time.
    poke = window[window['Event'].str.startswith("Poke:", na= False)] 
    
    # If there is no poke event in the window, raise an error. 
    # This should not happen, as the end time of the trial is defined as the time of the first poke event after the trial's start time. 
    # If it does happen, it indicates a problem with the data or the parsing logic.
    if poke.empty:
        raise ValueError("No poke event found in the window.")
    # Assuming there is only one poke event in the window, we can take the first row of the poke DataFrame.
    poke_event = poke['Event'].iloc[0]
    
    # Extract ChosenPort and CorrectPort from the poke_event string.
    chosen_port = int(poke_event.split("ChosenPort=")[1].split(",")[0])   
    correct_port = int(poke_event.split("CorrectPort=")[1].split("-")[0])
    # If there is no poke (ChosenPort = -1), return NaN.
    if chosen_port == -1:
        return np.nan
    # Calculate the angle offset as (ChosenPort - CorrectPort) * (360 / nosepoke_count).
    angle_offset = (chosen_port - correct_port) * (360 / nosepoke_count)
    # Adjust the angle offset to be between -180 and 180 degrees.
    return ((angle_offset + 180) % 360) - 180

def _get_target_zone_size(window: pd.DataFrame
) -> float:
    '''
    Return the target zone size for a trial, determined by the radius value in the "Target zone available*" log within a given window.

    Parameters
    ----------
    window : pd.DataFrame
        A DataFrame containing event logs for a trial, filtered to the time window between the trial's start time and end time.
    
    Returns
    -------
    float
        The target zone size for the trial, determined by the radius value in the "Target zone available*" log within the given window. 
        If there is no such log within the window, return NaN.
    '''
    # Get the target zone log from the window. This should be the first occurrence of a log starting with "Target zone available" after 
    # the trial's start time.
    tz= window[window['Event'].str.contains("Target zone available", na = False)]

    # If there is no target zone log in the window, return NaN.
    if tz.empty:
        return np.nan

    # If there is a target zone log, extract the radius value from the log string. The log string should be in the format 
    # "Target zone available (X Y : {int} {int} - Radius : {int})", so we can split the string to extract the radius value.
    radius_str = tz['Event'].iloc[0].split("Radius : ")[1].split(")")[0]
    return float(radius_str)







####################################################################################
#   Parse Events to Trials Main Function                                           #
####################################################################################

def parse_events_to_trials(events_df: pd.DataFrame = None,
                           nosepoke_count: int = 18,
        
) -> pd.DataFrame:
    '''
    Build a DataFrame with trial-level information from a DataFrame containing event logs.

    Parameters
    ----------
    events_df : pd.DataFrame
        A  `data-conduit` ExperimentEvents instance (i.e., ExperimentEvents.df) containing event logs for a session.
    nosepoke_count : int
        The number of nosepoke ports in the arena. This is used to calculate the angle offset.
    
    Returns
    -------
    pd.DataFrame
        One row per trial: trial_index, start_time, end_time, TTP,
        outcome, LED, angle_offset, target_zone_size.
    '''

    # Initialise an empty list to store trial information.
    trials = []
    # Initialise trial index and previous trial end time.
    previous_trial_end_time = None

    
    # Create a mask to identify rows in the events_df where the "Event" column starts with "Poke:", 
    # and use this mask to create a DataFrame containing only poke events.
    poke_mask = events_df["Event"].str.startswith("Poke:", na=False)
    poke_rows = events_df[poke_mask]


    # Iterate over the poke events to define trials. Each trial starts at the previous trial's end time + 1 second 
    # (or the "Start trial logic" log for the first trial), and ends at the time of the poke event.
    for i, (poke_time, poke_row) in enumerate(poke_rows.iterrows()):

        # Get the start time of the trial.
        start_time = _get_start_time(events_df, 
                                     i, 
                                     previous_trial_end_time
                                     )
        
        # The end time of the trial is the time of the current poke event.
        end_time = poke_time

        # Define a window of events between the trial's start time and end time.
        trial_window = _within_window(events_df, start_time, end_time)

        trials.append({
            'trial_index': i + 1,  # trial index starts at 1
            'start_time': start_time,
            'end_time': end_time,
            'TTP': _get_TTP(trial_window),
            'ChosenPort': int(poke_row['Event'].split("ChosenPort=")[1].split(",")[0]),
            'CorrectPort': int(poke_row['Event'].split("CorrectPort=")[1].split("-")[0]),
            'outcome': _get_outcome(poke_row['Event']),
            'LED': _get_LED(trial_window),
            'angle_offset': _get_angle_offset(trial_window, nosepoke_count),
            'target_zone_size': _get_target_zone_size(trial_window),
        })
        # Update the previous trial end time for the next iteration.
        previous_trial_end_time = end_time
    # Convert the list of trials to a DataFrame and return it.
    return pd.DataFrame(trials)


    

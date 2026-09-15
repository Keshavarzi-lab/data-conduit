'''
Miscellaneous functions for movement_figures.
---------------------------------------------





'''



import pandas as pd
from IPython.display import display, Markdown

from data_conduit.datastructures import DataObject, StreamMap


def visual_streams_check(loaded_ds_object,
                         show_streams: bool = True,
                         show_stream_shapes: bool = True,
                         show_stream_dtypes: bool = True,
                         display_streams: bool = True
                         ):
    '''
    Visual streams check for the loaded dataset object. 
    Returns summary table of streams and optionally displays the streams.

    Parameters
    ----------
    loaded_ds_object: 
        The loaded dataset object.
    show_streams (bool): 
        Whether to show streams.
    show_stream_shapes (bool): 
        Whether to show stream shapes.
    show_stream_dtypes (bool): 
        Whether to show stream data types.
    display_streams (bool): 
        Whether to display streams.

    Returns
    --------
    pd.DataFrame:
        Summary table of streams.

    '''


    #==== Check if the loaded_ds_object is a valid object
    if not isinstance(loaded_ds_object, dict):
        raise ValueError(f'''
                        The loaded_ds_object is not a valid StreamMap object.
                        Received type: {type(loaded_ds_object)}.
                        The loaded_ds_object must be obtained from `datastructure.load()` function.'''
        )
    
    




    summary_table = []
    for stream_key in loaded_ds_object:
        if show_streams:
            summary_table.append({
            'Stream Name': stream_key,
        })
        if show_stream_shapes:
            summary_table[-1]['Shape'] = loaded_ds_object[stream_key].shape if hasattr(loaded_ds_object[stream_key], 'shape') else (loaded_ds_object[stream_key].sizes if hasattr(loaded_ds_object[stream_key], 'sizes') else None)
        if show_stream_dtypes:
            summary_table[-1]['Type'] = type(loaded_ds_object[stream_key]).__name__


    display(Markdown("=========================\n### Summary of Loaded Streams\n========================="))
    display(pd.DataFrame(summary_table))

    if display_streams:
        display(Markdown("=========================\n### Loaded Streams\n========================="))
        for stream_key in loaded_ds_object:
            display(Markdown(f"#### Stream: {stream_key}"))
            display(loaded_ds_object[stream_key])

        

    


    
#  'start_inclusive', 
#  'end_inclusive', 
#  'first_event_index', 
#  'last_event_index', 
#  'target_zone_size', 

#  'outbound_start_inclusive', 
#  'outbound_end_inclusive', 
#  'inbound_start_inclusive', 
#  'inbound_end_inclusive',

default_keep = ['trial_index', 'start_time', 'end_time', 'tz_triggered_time', 'tz_available_time', 
                'ChosenPort', 'CorrectPort', 'outcome', 'LED', 'angle_offset', 'TTT', 'TTP', 
                'outbound_start_time', 'outbound_end_time', 'inbound_start_time', 
                'inbound_end_time', 'session', 'mouseID', 'day'
                ]
''' 
Contains
---------
- 'trial_index' 
- 'start_time'
- 'end_time'
- 'tz_triggered_time'
- 'tz_available_time'
- 'ChosenPort'
- 'CorrectPort'
- 'outcome'
- 'LED'
- 'angle_offset'
- 'TTT'
- 'TTP'
- 'outbound_start_time'
- 'outbound_end_time'
- 'inbound_start_time'
- 'inbound_end_time'
- 'session'
- 'mouseID'
- 'day'
'''

default_drop = ['start_inclusive', 
                'end_inclusive', 
                'first_event_index', 
                'last_event_index', 
                'target_zone_size', 
                'outbound_start_inclusive', 
                'outbound_end_inclusive', 
                'inbound_start_inclusive', 
                'inbound_end_inclusive']
'''Contains
---------
- 'start_inclusive'
- 'end_inclusive'
- 'first_event_index'
- 'last_event_index'
- 'target_zone_size'
- 'outbound_start_inclusive'
- 'outbound_end_inclusive'
- 'inbound_start_inclusive'
- 'inbound_end_inclusive'
'''



def configure_simplified_trial_df(trial_df: pd.DataFrame,                   # The trial dataframe to configure.
                                  keep_columns: list | None = default_keep,         # Columns to keep in the simplified trial dataframe. If None, all columns not specified in drop_columns will be kept.
                                  drop_columns: list | None = None,                 # Columns to drop from the simplified trial dataframe. If None, no columns will be dropped.
                                  modify_in_place: bool = False,            # Whether to modify the trial dataframe in place or return a new dataframe. If True, the original trial dataframe will be modified and returned. If False, a new dataframe will be returned.
                                  ) -> pd.DataFrame:
    '''
    Configure a simplified trial dataframe by keeping or dropping specified columns.

    Parameters
    ----------
    trial_df : pd.DataFrame
        The trial dataframe to configure.
    keep_columns : list, optional
        Columns to keep in the simplified trial dataframe. 
        If None, all columns not specified in drop_columns will be kept. Default is default_keep.
        Default is:
        ['trial_index', 'start_time', 'end_time', 'tz_triggered_time', 'tz_available_time', 
        'ChosenPort', 'CorrectPort', 'outcome', 'LED', 'angle_offset', 'TTT', 'TTP', 
        'outbound_start_time', 'outbound_end_time', 'inbound_start_time', 'inbound_end_time', 
        'session', 'mouseID', 'day']


    drop_columns : list, optional
        Columns to drop from the simplified trial dataframe. 
        If None, no columns will be dropped. Default is None.
        The predefined default_drop list is:
        ['start_inclusive', 'end_inclusive', 'first_event_index', 'last_event_index', 
        'target_zone_size', 'outbound_start_inclusive', 'outbound_end_inclusive', 
        'inbound_start_inclusive', 'inbound_end_inclusive']

    modify_in_place : bool, optional
        Whether to modify the trial dataframe in place or return a new dataframe. 
        If True, the original trial dataframe will be modified and returned. 
        If False, a new dataframe will be returned. Default is False.
    Returns
    -------
    pd.DataFrame
        The configured trial dataframe.



    '''
    
    if not modify_in_place:
        trial_df = trial_df.copy()

    if keep_columns is not None:
        trial_df.drop(columns=trial_df.columns.difference(keep_columns), inplace=True)
        # Move retained columns into the requested order on the same DataFrame.
        for column in reversed(keep_columns):
            trial_df.insert(0, column, trial_df.pop(column))

    if drop_columns is not None:
        trial_df.drop(columns=drop_columns, inplace=True)

    return trial_df

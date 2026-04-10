'''
Segment module core.
---------------------

Description:
    Global-time segmentation for data-conduit.

    One function: get_segment. Pass values for a lookup column,
    specify which column you want back. Returns an array you can
    pass straight to another DataFrame/DataArray to extract data.
'''


################################################################################
# Imports
################################################################################

import numpy as np
import pandas as pd

################################################################################




################################################################################
# get_segment
################################################################################

def get_segment(
    data: pd.DataFrame,
    start: float,
    end: float,
    lookup_column: str,
    return_column: str,
) -> np.ndarray:
    '''
    Get values from return_column where lookup_column is between start and end.

    Parameters
    ----------
    data : pd.DataFrame
        Table containing both columns.
    start : float
        Start of the range (inclusive) in lookup_column.
    end : float
        End of the range (inclusive) in lookup_column.
    lookup_column : str
        Column to filter on.
    return_column : str
        Column to return values from.

    Returns
    -------
    np.ndarray
        Matching values from return_column. Pass this directly to
        .sel(), .loc[], or another lookup to get data from a
        different stream.
    '''
    mask = (data[lookup_column] >= start) & (data[lookup_column] <= end)
    return data.loc[mask, return_column].values


################################################################################

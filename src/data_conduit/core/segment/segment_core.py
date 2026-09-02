"""
Segment module core.

Contents
--------
get_segment
    Extract a window of values from a DataFrame column using a range
    filter on a separate lookup column.
"""

################################################################################
# Imports
################################################################################
import numpy as np
import pandas as pd


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
    """Extract values from one column where another column falls within a range.

    Filters ``data`` to rows where ``lookup_column`` is between ``start`` and
    ``end`` (inclusive), then returns the corresponding values from
    ``return_column`` as a NumPy array.

    A typical use case is extracting global-clock index values that fall within
    a trial window, which can then be passed to ``.sel()`` or ``.loc[]`` to
    slice a DataArray or DataFrame aligned to the same global clock.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing both ``lookup_column`` and ``return_column``.
    start : float
        Start of the range (inclusive) applied to ``lookup_column``.
    end : float
        End of the range (inclusive) applied to ``lookup_column``.
    lookup_column : str
        Name of the column to filter on (e.g. a time or global-clock column).
    return_column : str
        Name of the column whose values are returned for the matching rows.

    Returns
    -------
    np.ndarray
        Values from ``return_column`` for all rows where ``lookup_column``
        is within ``[start, end]``. Pass this directly to ``.sel()``,
        ``.loc[]``, or another index lookup to extract the corresponding
        data from a separate stream aligned to the same column.

    Examples
    --------
    Extract global-clock indices covering a trial window, then use them to
    slice a DataArray:

    >>> indices = get_segment(
    ...     data=index_map,
    ...     start=trial_start,
    ...     end=trial_end,
    ...     lookup_column="GlobalTime",
    ...     return_column="Index",
    ... )
    >>> trial_data = data_array.sel(Time=indices)
    """
    mask = (data[lookup_column] >= start) & (data[lookup_column] <= end)
    return data.loc[mask, return_column].values


################################################################################

"""Extract inclusive windows from tabular, time-aligned data."""

import numpy as np
import pandas as pd


def get_segment(
    data: pd.DataFrame,
    start: float,
    end: float,
    lookup_column: str,
    return_column: str,
) -> np.ndarray:
    """Return values whose lookup column lies within ``[start, end]``.

    The returned NumPy array can be passed directly to ``.sel()``, ``.loc[]``,
    or another index lookup for a stream aligned to ``return_column``.

    Examples
    --------
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

'''
Utilities to assist in quickly inspecting sessions and trials.

Contents
--------
- group_trials
    Collapse a SessionSettings trials table so identical trial settings are
    shown once, with a list of trial labels for the rows that share them.
- trial_column_summary
    Report which trial settings columns vary and which are constant.
- compare_trial_settings
    Compare settings between two trials, groups, or user-selected trial sets.
'''

#===== Imports

from collections.abc import Hashable
from typing import Any

import pandas as pd


def group_trials(
        data: Any,
        *,
        include_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        varying_only: bool | str = 'unchanged',             # options: 'unchanged', 'discard_matching', 'keep_matching'
        include_group_index: bool = False,
        group_column: str = 'group',
        trial_column: str = 'trial',
    ) -> pd.DataFrame:
    '''
    Collapse a per-trial SessionSettings table into groups of matching settings.

    SessionSettings stores one row per trial. That is useful when you need the
    exact trial-by-trial table, but it can be noisy when many trials have the
    same settings. This function groups rows that have the same values across
    the selected settings columns, then returns one row per unique settings
    combination. The output includes a leading ``trial_column`` containing a
    list such as ``['trial_1', 'trial_2', 'trial_3']`` for the trials in that
    group.

    The input can be:
    - a SessionSettings object, where ``obj.df['trials']`` is used;
    - a split SessionSettings data dict, where ``data['trials']`` is used;
    - a trials DataFrame directly.

    By default, all columns are considered except obvious trial labels. A
    DataFrame index or first column containing values like ``trial_1``,
    ``trial_2``, ... is treated as the label source rather than a setting to
    compare. A first column named ``trial`` is handled the same way. Those
    labels are preserved in the output ``trial_column`` list.

    ``varying_only`` controls how columns that match across all collapsed
    groups are displayed:
    - ``'discard_matching'`` or ``True`` keeps only columns that differ between
      groups. For example, ``maxRuntime`` is omitted if it is identical for
      every group.
    - ``'keep_matching'`` keeps only columns that are identical across every
      group.
    - ``'unchanged'`` or ``False`` keeps both matching and differing columns.

    Set ``include_group_index=True`` to add a leading ``group`` column with
    labels like ``group_1``, ``group_2``, etc. This is useful when you want to
    refer back to collapsed rows in ``compare_trial_settings``.

    Parameters
    ----------
    data : Any
        SessionSettings object, ``{'metadata': df, 'trials': df}`` dict, or
        trials DataFrame.
    include_columns : list[str], optional
        Columns to use as the grouping criteria. If omitted, every non-label
        column is used unless excluded by ``exclude_columns``.
    exclude_columns : list[str], optional
        Columns to ignore when deciding whether trials have the same settings.
        This is useful for fields that naturally vary but are not relevant to
        the comparison you are doing.
    varying_only : bool or {'unchanged', 'discard_matching', 'keep_matching'}, default 'unchanged'
        Controls whether columns that match across all collapsed groups are
        discarded, exclusively kept, or left unchanged. This affects display
        only; grouping is still performed using the selected ``include_columns``
        / ``exclude_columns`` criteria. ``True`` is treated as
        ``'discard_matching'`` for backward compatibility, and ``False`` is
        treated as ``'unchanged'``.
    include_group_index : bool, default False
        If True, add a leading group-label column.
    group_column : str, default 'group'
        Name of the optional group-label column.
    trial_column : str, default 'trial'
        Name of the output column that stores the list of trial labels. 
        This column is always included in the output, and the remaining columns 
        are the settings columns used for grouping. 
        

    Returns
    -------
    pd.DataFrame
        One row per unique settings combination. The first column is
        ``trial_column`` and contains lists of trial labels. The remaining
        columns are the settings columns used for grouping.

    Examples
    --------
    Group a SessionSettings object using all settings columns:

    >>> grouped = group_trials(ss)

    Ignore columns that should not split otherwise identical trials:

    >>> grouped = group_trials(ss, exclude_columns=['seconds'])

    Only compare a small set of columns:

    >>> grouped = group_trials(trials_df, include_columns=['rewardTimeout', 'maskingSound.play'])

    Focus the output on settings that differ between groups:

    >>> grouped = group_trials(ss, exclude_columns=['seconds'], varying_only='discard_matching')

    Show only settings that are identical across groups:

    >>> grouped = group_trials(ss, exclude_columns=['seconds'], varying_only='keep_matching')

    Add group labels for later comparison:

    >>> grouped = group_trials(ss, include_group_index=True)
    '''
    trials_df = _get_trials_dataframe(data)
    label_column = _detect_trial_label_column(trials_df)

    exclude = set(exclude_columns or [])
    if label_column is not None:
        exclude.add(label_column)

    if include_columns is None:
        group_columns = [col for col in trials_df.columns if col not in exclude]
    else:
        missing = [col for col in include_columns if col not in trials_df.columns]
        if missing:
            raise KeyError(f"Columns not found in trials DataFrame: {missing}")
        group_columns = [col for col in include_columns if col not in exclude]

    if not group_columns:
        raise ValueError("No columns left to group by after applying include/exclude choices.")

    trial_labels = _trial_labels(trials_df, label_column)

    groups: dict[tuple[Hashable, ...], dict[str, Any]] = {}
    for row_position, (_, row) in enumerate(trials_df.iterrows()):
        key = tuple(_make_hashable(row[col]) for col in group_columns)

        if key not in groups:
            # Store display values from the first row in each group. Later rows
            # with the same key only add their trial label.
            groups[key] = {
                trial_column: [],
                **{col: row[col] for col in group_columns},
            }

        groups[key][trial_column].append(trial_labels[row_position])

    grouped = pd.DataFrame(groups.values(), columns=[trial_column, *group_columns])

    if include_group_index:
        grouped.insert(
            0,
            group_column,
            [f'group_{group_number}' for group_number in range(1, len(grouped) + 1)],
        )

    display_mode = _normalise_varying_only(varying_only)
    if display_mode != 'unchanged':
        label_columns = [trial_column]
        if include_group_index:
            label_columns.insert(0, group_column)
        if display_mode == 'discard_matching':
            display_columns = _varying_columns(grouped, exclude_columns=label_columns)
        else:
            display_columns = _matching_columns(grouped, exclude_columns=label_columns)
        grouped = grouped.loc[:, [*label_columns, *display_columns]]

    return grouped


def compare_trial_settings(
        data: Any,
        left: Any,
        right: Any,
        *,
        include_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        trial_column: str = 'trial',
        group_column: str = 'group',
        show: str = 'all',                  # options: 'all', 'different', 'same'
    ) -> pd.DataFrame:
    '''
    Compare settings between two trials, groups, or trial sets.

    This function answers the practical question: "what is actually the same
    or different between these two things?" It returns one row per inspected
    setting column with the value on the left, the value on the right, and an
    ``is_same`` flag.

    The input can be a raw SessionSettings object/data dict/trials DataFrame, or
    a collapsed table from ``group_trials``. The ``left`` and ``right``
    selectors can be:
    - trial labels such as ``'trial_1'``;
    - group labels such as ``'group_2'`` when the grouped table has a group
      column;
    - integer row positions;
    - lists of any of the above, e.g. ``['trial_1', 'trial_2']``.

    If a selector refers to multiple rows, each setting is summarised. A setting
    that is identical across the selected rows is shown as a single value; a
    setting that varies within the selected rows is shown as a list of observed
    values.

    Parameters
    ----------
    data : Any
        SessionSettings object, ``{'metadata': df, 'trials': df}`` dict, raw
        trials DataFrame, or grouped DataFrame from ``group_trials``.
    left, right : Any
        Selections to compare. Use trial labels, group labels, row positions, or
        lists of those selectors.
    include_columns : list[str], optional
        Columns to compare. If omitted, all non-label columns are compared.
    exclude_columns : list[str], optional
        Columns to ignore.
    trial_column : str, default 'trial'
        Column containing trial labels or trial-label lists.
    group_column : str, default 'group'
        Column containing optional group labels.
    show : {'all', 'different', 'same'}, default 'all'
        Filter the comparison output.

    Returns
    -------
    pd.DataFrame
        Columns are ``setting``, ``left``, ``right``, and ``is_same``.

    Examples
    --------
    Compare two raw trials:

    >>> compare_trial_settings(session_data['trials'], 'trial_1', 'trial_8')

    Compare two collapsed groups:

    >>> grouped = group_trials(ss, include_group_index=True)
    >>> compare_trial_settings(grouped, 'group_1', 'group_2', show='different')

    Compare two user-defined sets of trials:

    >>> compare_trial_settings(ss, ['trial_1', 'trial_2'], ['trial_9', 'trial_10'])
    '''
    if show not in {'all', 'different', 'same'}:
        raise ValueError("show must be one of: 'all', 'different', 'same'.")

    df = _get_trials_dataframe(data)
    compare_columns = _comparison_columns(
        df,
        include_columns=include_columns,
        exclude_columns=exclude_columns,
        trial_column=trial_column,
        group_column=group_column,
    )

    left_values = _selection_values(
        df,
        left,
        columns=compare_columns,
        trial_column=trial_column,
        group_column=group_column,
    )
    right_values = _selection_values(
        df,
        right,
        columns=compare_columns,
        trial_column=trial_column,
        group_column=group_column,
    )

    left_label = _selector_label(left)
    right_label = _selector_label(right)

    rows = []
    for col in compare_columns:
        left_value = left_values[col]
        right_value = right_values[col]
        is_same = _make_hashable(left_value) == _make_hashable(right_value)
        rows.append({
            'setting': col,
            left_label: left_value,
            right_label: right_value,
            'is_same': is_same,
        })

    result = pd.DataFrame(rows)
    if show == 'different':
        result = result.loc[~result['is_same']]
    elif show == 'same':
        result = result.loc[result['is_same']]

    return result.reset_index(drop=True)


def trial_column_summary(
        data: Any,
        *,
        include_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> pd.DataFrame:
    '''
    Summarise which SessionSettings trial columns vary and which are constant.

    This is useful before or after grouping. Columns with ``n_unique == 1`` are
    not helpful for distinguishing trial settings groups, while columns with
    larger ``n_unique`` values explain where the trial settings actually differ.

    Parameters
    ----------
    data : Any
        SessionSettings object, ``{'metadata': df, 'trials': df}`` dict, or
        trials DataFrame.
    include_columns : list[str], optional
        Columns to inspect. If omitted, all non-label columns are inspected.
    exclude_columns : list[str], optional
        Columns to ignore.

    Returns
    -------
    pd.DataFrame
        One row per inspected column with ``n_unique``, ``is_constant``, and a
        short list of observed values.
    '''
    trials_df = _get_trials_dataframe(data)
    label_column = _detect_trial_label_column(trials_df)

    exclude = set(exclude_columns or [])
    if label_column is not None:
        exclude.add(label_column)

    if include_columns is None:
        columns = [col for col in trials_df.columns if col not in exclude]
    else:
        missing = [col for col in include_columns if col not in trials_df.columns]
        if missing:
            raise KeyError(f"Columns not found in trials DataFrame: {missing}")
        columns = [col for col in include_columns if col not in exclude]

    rows = []
    for col in columns:
        values = []
        seen = set()
        for value in trials_df[col]:
            key = _make_hashable(value)
            if key not in seen:
                seen.add(key)
                values.append(value)

        rows.append({
            'column': col,
            'n_unique': len(seen),
            'is_constant': len(seen) == 1,
            'values': values[:5],
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    return summary.sort_values(['is_constant', 'n_unique', 'column']).reset_index(drop=True)


def _comparison_columns(
        df: pd.DataFrame,
        *,
        include_columns: list[str] | None,
        exclude_columns: list[str] | None,
        trial_column: str,
        group_column: str,
    ) -> list[str]:
    '''Resolve which columns should be compared between selections.'''
    label_column = _detect_trial_label_column(df)
    exclude = set(exclude_columns or [])

    for col in (label_column, trial_column, group_column):
        if col is not None:
            exclude.add(col)

    if include_columns is None:
        return [col for col in df.columns if col not in exclude]

    missing = [col for col in include_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    return [col for col in include_columns if col not in exclude]


def _selection_values(
        df: pd.DataFrame,
        selector: Any,
        *,
        columns: list[str],
        trial_column: str,
        group_column: str,
    ) -> dict[str, Any]:
    '''
    Return representative values for a trial/group selection.

    Single-row selections return that row's values. Multi-row selections return
    a scalar when all selected rows agree, otherwise a list of observed values.
    '''
    rows = _selection_rows(
        df,
        selector,
        trial_column=trial_column,
        group_column=group_column,
    )

    values = {}
    for col in columns:
        unique_values = []
        seen = set()
        for value in rows[col]:
            key = _make_hashable(value)
            if key not in seen:
                seen.add(key)
                unique_values.append(value)

        values[col] = unique_values[0] if len(unique_values) == 1 else unique_values

    return values


def _selection_rows(
        df: pd.DataFrame,
        selector: Any,
        *,
        trial_column: str,
        group_column: str,
    ) -> pd.DataFrame:
    '''Resolve a selector into one or more DataFrame rows.'''
    if isinstance(selector, int):
        return df.iloc[[selector]]

    if isinstance(selector, (list, tuple, set)):
        parts = [
            _selection_rows(
                df,
                item,
                trial_column=trial_column,
                group_column=group_column,
            )
            for item in selector
        ]
        return pd.concat(parts, axis=0)

    if isinstance(selector, str):
        if group_column in df.columns:
            matches = df[group_column].astype(str) == selector
            if matches.any():
                return df.loc[matches]

        if trial_column in df.columns:
            matches = df[trial_column].apply(
                lambda value: (
                    selector in [str(item) for item in value]
                    if isinstance(value, list)
                    else str(value) == selector
                )
            )
            if matches.any():
                return df.loc[matches]

        if selector in df.index.astype(str):
            return df.loc[[idx for idx in df.index if str(idx) == selector]]

    raise KeyError(f"Selection not found: {selector!r}")


def _selector_label(selector: Any) -> str:
    '''Return a short human-readable column name for a left/right selector.'''
    if isinstance(selector, str):
        return selector
    if isinstance(selector, int):
        return f'row_{selector}'
    if isinstance(selector, (list, tuple)):
        return ', '.join(_selector_label(item) for item in selector)
    return str(selector)


def _get_trials_dataframe(data: Any) -> pd.DataFrame:
    '''
    Resolve the caller's input into the trials DataFrame.

    This keeps ``group_trials`` convenient in notebooks: pass the
    SessionSettings object, its ``.df`` dict, or the DataFrame itself.
    '''
    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, dict) and isinstance(data.get('trials'), pd.DataFrame):
        return data['trials']

    df_attr = getattr(data, 'df', None)
    if isinstance(df_attr, dict) and isinstance(df_attr.get('trials'), pd.DataFrame):
        return df_attr['trials']

    raise TypeError(
        "data must be a trials DataFrame, a {'trials': DataFrame} dict, "
        "or an object with .df['trials']."
    )


def _detect_trial_label_column(df: pd.DataFrame) -> str | None:
    '''
    Identify an existing trial-label column that should not be grouped on.

    SessionSettings now adds ``trial`` explicitly, but this also handles any
    first column whose values already look like trial labels.
    '''
    if df.empty or len(df.columns) == 0:
        return None

    first_column = df.columns[0]
    if first_column == 'trial':
        return first_column

    first_values = df[first_column].dropna().astype(str)
    if first_values.empty:
        return None

    if first_values.str.fullmatch(r'trial_\d+').all():
        return first_column

    return None


def _trial_labels(df: pd.DataFrame, label_column: str | None) -> list[str]:
    '''Return visible trial labels, falling back to trial_1, trial_2, ... .'''
    if label_column is not None:
        return df[label_column].astype(str).tolist()

    index_labels = df.index.to_series().dropna().astype(str)
    if not index_labels.empty and index_labels.str.fullmatch(r'trial_\d+').all():
        return df.index.astype(str).tolist()

    return [f'trial_{trial_number}' for trial_number in range(1, len(df) + 1)]


def _varying_columns(df: pd.DataFrame, *, exclude_columns: list[str]) -> list[str]:
    '''Return columns with more than one distinct value.'''
    exclude = set(exclude_columns)
    varying = []

    for col in df.columns:
        if col in exclude:
            continue

        unique_values = {_make_hashable(value) for value in df[col]}
        if len(unique_values) > 1:
            varying.append(col)

    return varying


def _matching_columns(df: pd.DataFrame, *, exclude_columns: list[str]) -> list[str]:
    '''Return columns with one distinct value across all rows.'''
    exclude = set(exclude_columns)
    matching = []

    for col in df.columns:
        if col in exclude:
            continue

        unique_values = {_make_hashable(value) for value in df[col]}
        if len(unique_values) == 1:
            matching.append(col)

    return matching


def _normalise_varying_only(value: bool | str) -> str:
    '''Map the display-mode option onto one internal vocabulary.'''
    if value is True:
        return 'discard_matching'
    if value is False:
        return 'unchanged'

    valid_modes = {'unchanged', 'discard_matching', 'keep_matching'}
    if value not in valid_modes:
        raise ValueError(
            "varying_only must be True, False, or one of "
            f"{sorted(valid_modes)}."
        )

    return value


def _make_hashable(value: Any) -> Hashable:
    '''
    Convert cell values into stable group keys.

    Pandas groupby can struggle with lists/dicts in cells. This recursive
    conversion lets us compare those values if they appear in a trials table.
    Missing values are normalised so separate NaN objects group together.
    '''
    if isinstance(value, list):
        return tuple(_make_hashable(item) for item in value)

    if isinstance(value, tuple):
        return tuple(_make_hashable(item) for item in value)

    if isinstance(value, dict):
        return tuple(
            (key, _make_hashable(val))
            for key, val in sorted(value.items())
        )

    if pd.isna(value):
        return '<NA>'

    return value


__all__ = [
    'compare_trial_settings',
    'group_trials',
    'trial_column_summary',
]

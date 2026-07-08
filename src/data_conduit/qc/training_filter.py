'''
Training-session selection and trial filtering from Celia's detail spreadsheet.
==============================================================================

Description:
    Celia's ``Training-trials_detail.xlsx`` records, per mouse and per training
    "day", WHICH Bonsai session(s) to analyse and WHICH trials to keep from each.
    That spreadsheet is small and effectively static, so its contents are
    transcribed HERE as the hardcoded ``TRAINING_DETAIL`` table (edit it directly if
    the spreadsheet changes) and fed through two independent steps, kept separate on
    purpose so the raw trial table can be inspected between them:

      1. SESSION SELECTION (``training_spec`` + ``training_session_names``): flatten
         the hardcoded table into a tidy spec (one row per mouse+session) and derive
         the flat allowlist of session-folder names to hand to
         ``qc_datastructure(..., include=...)``. Loading with that allowlist gives
         the FULL, UNFILTERED trial table for exactly the training sessions of
         interest.

      2. TRIAL FILTERING (``filter_trials``, previewed by ``summarise_training_filter``):
         apply the per-session rules to the loaded trial table. Every session keeps
         ``trial_index >= min_trial`` (the "X to end" rule). The two block-protocol
         mice (MR / MbR) additionally drop the mid-session cue trials, which the
         spreadsheet annotates as "exclude N ON trial"; those are exactly the
         ``LED == 'ON'`` trials in the kept range, so the filter drops all of them
         rather than trusting the hand-written count (sessions that ran short or
         over-ran contain fewer / more ON blocks than the annotation predicted).

    Why match on the SESSION name and not the day: the spreadsheet's day numbers do
    NOT line up with the on-disk ``Day<N>`` folders (e.g. FL "day 2" is the Bonsai
    file ``2026-05-11T125643Z`` which lives on disk under ``Day7``). The Bonsai
    session filename is the only reliable key, so both steps join on it.

    Combined days: some days pool two or three sessions ("COMBINE ... SESSIONS").
    Each pooled session carries its own ``min_trial``, so pooling needs no special
    case here; the table simply gives the pooled sessions the same ``training_day``,
    and ``filter_trials`` renumbers the kept trials 1..N within each
    (mouse, training_day) as ``day_trial_index`` so a combined day reads as one
    continuous sequence.

Contents:
--------------------------------
- TRAINING_DETAIL:            Hardcoded transcription of the spreadsheet.
- training_spec:              Hardcoded table -> tidy per-session spec DataFrame.
- training_session_names:     Spec -> flat session-name allowlist for ``include=``.
- summarise_training_filter:  Preview what ``filter_trials`` will keep / drop.
- filter_trials:              Apply the per-session cut + ON-exclusion rules.
'''




################################################################################
# Imports
################################################################################

import pandas as pd

################################################################################




################################################################################
# Hardcoded spreadsheet transcription
################################################################################

# Transcribed from ``Training-trials_detail.xlsx`` (Celia). Edit here if the
# spreadsheet changes. Keyed by mouse; each entry is one Bonsai session:
#
#     (training_day, session, min_trial, exclude_on)
#
#   * training_day : the spreadsheet's day number. NOT the on-disk Day<N> folder;
#                    a day may list two or three sessions ("COMBINE ... SESSIONS"),
#                    which simply repeat the same training_day.
#   * session      : the Bonsai session-folder name (the reliable join key).
#   * min_trial    : keep trial_index >= min_trial (the "X to end" rule).
#   * exclude_on   : also drop this session's LED=='ON' cue trials. True only for
#                    the two block-protocol mice (MR_M01569515, MbR_M01569518),
#                    where the spreadsheet says "+ exclude N ON trial".
#
# Days with no session listed in the spreadsheet (e.g. FL / MR "day 1") are simply
# absent below.

# TRAINING_DETAIL: dict[str, list[tuple[int, str, int, bool]]] = {
#     'FL_M01569519': [
#         (1, '2026-04-21T144313Z', 16, False),
#         (1, '2026-05-12T164637Z', 24, False),
#         (4, '2026-05-13T161503Z', 24, False),
#         (5, '2026-05-14T165819Z', 24, False),
#         # (2, '2026-05-11T125643Z', 16, False),
#         # (3, '2026-05-12T164637Z', 24, False),
#         # (4, '2026-05-13T161503Z', 24, False),
#         # (5, '2026-05-14T165819Z', 24, False),
#     ],
#     'FLR_M01569521': [
#         (1, '2026-04-23T160907Z', 11, False),
#         (2, '2026-04-25T190120Z', 11, False),   # day 2: COMBINE TWO SESSIONS
#         (2, '2026-04-25T190915Z', 11, False),
#         (3, '2026-04-28T124013Z',  6, False),   # day 3: COMBINE TWO SESSIONS
#         (3, '2026-04-28T124547Z', 11, False),
#         (4, '2026-04-29T161826Z', 11, False),
#         (5, '2026-04-30T151438Z',  6, False),
#     ],
#     'FbR_M01569522': [
#         (1, '2026-04-21T155201Z', 26, False),   # day 1: COMBINE TWO SESSIONS
#         (1, '2026-04-21T162256Z',  6, False),
#         (2, '2026-04-23T165237Z', 11, False),
#         (3, '2026-04-25T181535Z', 11, False),   # day 3: COMBINE TWO SESSIONS
#         (3, '2026-04-25T182521Z', 11, False),
#         (4, '2026-04-28T163449Z', 21, False),   # day 4: COMBINE THREE SESSIONS
#         (4, '2026-04-28T164540Z', 12, False),
#         (4, '2026-04-28T165257Z', 16, False),
#         (5, '2026-04-29T153239Z', 23, False),   # day 5: COMBINE TWO SESSIONS
#         (5, '2026-04-29T155431Z', 14, False),
#     ],
#     'MbL_M01569517': [
#         (1, '2026-05-20T112502Z', 24, False),
#         (2, '2026-05-21T120827Z', 24, False),
#         (3, '2026-05-22T150830Z', 24, False),
#         (4, '2026-05-28T154840Z', 24, False),
#         (5, '2026-05-29T155941Z', 24, False),
#     ],
#     # Block-protocol mice: keep 24-to-end AND drop the mid-session ON (cue) trials.
#     'MR_M01569515': [
#         (2, '2026-05-12T150308Z', 24, True),
#         (3, '2026-05-13T121852Z', 24, True),
#         (4, '2026-05-14T122109Z', 24, True),
#         (5, '2026-05-16T173516Z', 24, True),
#     ],
#     'MbR_M01569518': [
#         (1, '2026-05-14T135225Z', 24, True),
#         (2, '2026-05-16T155926Z', 24, True),
#         (3, '2026-05-17T154506Z', 24, True),
#         (4, '2026-05-19T110125Z', 24, True),
#         (5, '2026-05-20T102711Z', 24, True),
#     ],
# }

################################################################################
TRAINING_DETAIL: dict[str, list[tuple[int, str, int, bool]]] = {
    'FL_M01569519': [
        (1, '2026-04-25T194933Z', 11, False),
        (1, '2026-04-25T200011Z', 11, False),
        (1, '2026-04-25T200705Z', 11, False),
        (1, '2026-04-25T201616Z', 11, False),
        (1, '2026-04-25T202109Z', 6, False),
        (2, '2026-04-28T172539Z', 21, False),
        (2, '2026-04-28T173816Z', 6, False),
        (2, '2026-04-28T174249Z', 6, False),
        (2, '2026-04-28T174922Z', 2, False),
        (3, '2026-04-29T170501Z', 11, True),
        (3, '2026-04-29T171313Z', 11, False),
        (3, '2026-04-29T172239Z', 11, False),
        (3, '2026-04-29T174637Z', 2, False),
        (4, '2026-05-11T125643Z', 16, False),
        (5, '2026-05-12T164637Z', 24, False),
        # (6, '2026-05-13T161503Z', 24, False),
        # (7, '2026-05-14T165819Z', 24, False),
    ],
    'FLR_M01569521': [
        (1, '2026-04-23T160907Z', 11, False),
        (2, '2026-04-25T190120Z', 11, False),   # day 2: COMBINE TWO SESSIONS
        (2, '2026-04-25T190915Z', 11, False),
        (3, '2026-04-28T124013Z',  6, False),   # day 3: COMBINE TWO SESSIONS
        (3, '2026-04-28T124547Z', 11, False),
        (4, '2026-04-29T161826Z', 11, False),
        (5, '2026-04-30T151438Z',  6, False),
    ],
    'FbR_M01569522': [
        (1, '2026-04-21T155201Z', 26, False),   # day 1: COMBINE TWO SESSIONS
        (1, '2026-04-21T162256Z',  6, False),
        (2, '2026-04-23T165237Z', 11, False),
        (3, '2026-04-25T181535Z', 11, False),   # day 3: COMBINE TWO SESSIONS
        (3, '2026-04-25T182521Z', 11, False),
        (4, '2026-04-28T163449Z', 21, False),   # day 4: COMBINE THREE SESSIONS
        (4, '2026-04-28T164540Z', 12, False),
        (4, '2026-04-28T165257Z', 16, False),
        (5, '2026-04-29T153239Z', 23, False),   # day 5: COMBINE TWO SESSIONS
        (5, '2026-04-29T155431Z', 14, False),
    ],
    'MbL_M01569517': [
        (1, '2026-05-16T165921Z', 24, False),
        (2, '2026-05-17T144445Z', 24, False),
        (3, '2026-05-19T091455Z', 24, False),
        (4, '2026-05-20T112502Z', 24, False),
        (5, '2026-05-21T120827Z', 24, False),
        (6, '2026-05-22T150830Z', 24, False),
        (7, '2026-05-28T154840Z', 24, False),
        (8, '2026-05-29T155941Z', 24, False),
    ],
    # Block-protocol mice: keep 24-to-end AND drop the mid-session ON (cue) trials.
    'MR_M01569515': [
        (2, '2026-05-12T150308Z', 24, True),
        (3, '2026-05-13T121852Z', 24, True),
        (4, '2026-05-14T122109Z', 24, True),
        (5, '2026-05-16T173516Z', 24, True),
    ],
    'MbR_M01569518': [
        (1, '2026-05-12T132304Z', 24, True),
        (2, '2026-05-13T135855Z', 24, True),
        (3, '2026-05-14T135225Z', 24, True),
        (4, '2026-05-16T155926Z', 24, True),
        (5, '2026-05-17T154506Z', 24, True),
        (6, '2026-05-19T110125Z', 24, True),
        (7, '2026-05-20T102711Z', 24, True),
    ],
}

# TRAINING_DETAIL: dict[str, list[tuple[int, str, int, bool]]] = {
#     'FL_M01569519': [
#         (1, '2026-04-23T151952Z', 11, False),
#         (1, '2026-04-23T152710Z', 11, False),
#         (1, '2026-04-23T153735Z', 6, False),
#         (2, '2026-04-25T194933Z', 11, False),
#         (2, '2026-04-25T200011Z', 11, False),
#         (2, '2026-04-25T200705Z', 11, False),
#         (2, '2026-04-25T201616Z', 11, False),
#         (2, '2026-04-25T202109Z', 6, False),
#         (3, '2026-04-28T172539Z', 21, False),
#         (3, '2026-04-28T173816Z', 6, False),
#         (3, '2026-04-28T174249Z', 6, False),
#         (3, '2026-04-28T174922Z', 2, False),
#         (4, '2026-04-29T170501Z', 11, True),
#         (4, '2026-04-29T171313Z', 11, False),
#         (4, '2026-04-29T172239Z', 11, False),
#         (4, '2026-04-29T174637Z', 2, False),
#     ],
#     'MbL_M01569517': [
#         (1, '2026-04-25T161723Z', 11, False),
#         (1, '2026-04-25T163456Z', 11, False),
#         (2, '2026-04-26T144900Z', 11, False),
#         (3, '2026-04-27T132909Z', 11, False),
#         (3, '2026-04-27T134004Z', 11, False),
#         (3, '2026-04-27T135152Z', 11, False),
#         (4, '2026-04-28T104553Z', 11, False),
#         (4, '2026-04-28T105504Z', 11, False),
#         (5, '2026-04-29T124453Z', 11, False),
#         (5, '2026-04-29T125524Z', 4, False),
#         (5, '2026-04-29T130235Z', 4, False),
#         (5, '2026-04-29T130649Z', 4, False),
#         (6, '2026-05-07T093236Z', 16, False),
#         (7, '2026-05-08T111319Z', 16, False),
#     ],
#     'MbR_M01569518': [
#         (1, '2026-04-25T144913Z', 11, True),
#         (1, '2026-04-25T150252Z', 11, True),
#         (1, '2026-04-25T151326Z', 11, True),
#         (2, '2026-04-26T163755Z', 11, True),
#         (2, '2026-04-26T164301Z', 11, True),
#         (2, '2026-04-26T165604Z', 11, True),
#         (2, '2026-04-26T170917Z', 11, True),
#         (3, '2026-04-27T141618Z', 11, True),
#         (3, '2026-04-27T142320Z', 11, True),
#         (4, '2026-04-28T100411Z', 11, True),
#         (5, '2026-04-29T113619Z', 11, True),
#         (5, '2026-04-29T114758Z', 4, True),
#         (5, '2026-04-29T115844Z', 4, True),
#         (5, '2026-04-29T120528Z', 4, True),
#     ],
# }

# First / middle / last training day per mouse. ``training_day`` here is the PHASE
# RANK -- 1 = first, 2 = mid, 3 = last (see ``TRAINING_PHASE_LABELS``) -- NOT the
# calendar day; any inline ``# day N`` comments below refer to the original calendar
# day from Celia's sheet and are kept only for provenance. A SEPARATE curation from
# the 5-day progression above -- it includes earlier Day-1 sessions the 5-day list
# omits -- used by the phase (first/mid/last) plots. Reconstructed 2026-07-07 from the
# working-copy table that had been mislabelled ``TRAINING_DETAIL``.
TRAINING_FIRST_MID_LAST_DETAIL: dict[str, list[tuple[int, str, int, bool]]] = {
    'FL_M01569519': [
        (1, '2026-04-25T194933Z', 11, False),
        (1, '2026-04-25T200011Z', 11, False),
        (1, '2026-04-25T200705Z', 11, False),
        (1, '2026-04-25T201616Z', 11, False),
        (1, '2026-04-25T202109Z', 6, False),
        # (2, '2026-04-28T172539Z', 21, False),
        # (2, '2026-04-28T173816Z', 6, False),
        # (2, '2026-04-28T174249Z', 6, False),
        # (2, '2026-04-28T174922Z', 2, False),
        (2, '2026-04-29T170501Z', 11, True),
        (2, '2026-04-29T171313Z', 11, False),
        (2, '2026-04-29T172239Z', 11, False),
        (2, '2026-04-29T174637Z', 2, False),
        (3, '2026-05-11T125643Z', 16, False),
        # (5, '2026-05-12T164637Z', 24, False),


        # (1, '2026-04-25T194933Z', 11, False),   # first
        # (1, '2026-04-25T200011Z', 11, False),   # first  
        # (1, '2026-04-25T200705Z', 11, False),    # first  
        # (1, '2026-04-25T201616Z', 11, False),    # first 
        # (1, '2026-04-25T202109Z', 6, False),    # first 
        # (2, '2026-05-11T125643Z', 16, False),   # middle
        # (3, '2026-05-14T165819Z', 24, False),   # last
    ],
    'FLR_M01569521': [
        (1, '2026-04-23T160907Z', 11, False),
        # (2, '2026-04-25T190120Z', 11, False),   # day 2: COMBINE TWO SESSIONS
        # (2, '2026-04-25T190915Z', 11, False),
        (2, '2026-04-28T124013Z',  6, False),   # day 3: COMBINE TWO SESSIONS
        (2, '2026-04-28T124547Z', 11, False),
        (3, '2026-04-29T161826Z', 11, False),
        # (5, '2026-04-30T151438Z',  6, False),
    
    #     (1, '2026-04-23T160907Z', 11, False),   # first
    #     (2, '2026-04-28T124013Z',  6, False),   # middle
    #     (2, '2026-04-28T124547Z', 11, False),   # middle
    #     (3, '2026-04-30T151438Z',  6, False),   # last
    ],
    'FbR_M01569522': [
        (1, '2026-04-21T155201Z', 26, False),   # day 1: COMBINE TWO SESSIONS
        (1, '2026-04-21T162256Z',  6, False),
        # (2, '2026-04-23T165237Z', 11, False),
        # (3, '2026-04-25T181535Z', 11, False),   # day 3: COMBINE TWO SESSIONS
        # (3, '2026-04-25T182521Z', 11, False),
        (2, '2026-04-28T163449Z', 21, False),   # day 4: COMBINE THREE SESSIONS
        (3, '2026-04-28T164540Z', 12, False),
        (3, '2026-04-28T165257Z', 16, False),
        # (5, '2026-04-29T153239Z', 23, False),   # day 5: COMBINE TWO SESSIONS
        # (5, '2026-04-29T155431Z', 14, False),

        # (1, '2026-04-21T155201Z', 26, False),   # first: COMBINE TWO SESSIONS
        # (1, '2026-04-21T162256Z',  6, False),
        # (2, '2026-04-25T181535Z', 11, False),   # middle: COMBINE TWO SESSIONS
        # (2, '2026-04-25T182521Z', 11, False),
        # (3, '2026-04-29T153239Z', 23, False),   # last: COMBINE TWO SESSIONS
        # (3, '2026-04-29T155431Z', 14, False),
    ],
    'MbL_M01569517': [
        (1, '2026-05-16T165921Z', 24, False),
        # (2, '2026-05-17T144445Z', 24, False),
        # (3, '2026-05-19T091455Z', 24, False),
        (2, '2026-05-20T112502Z', 24, False),   # day 4
        # (5, '2026-05-21T120827Z', 24, False),
        # (6, '2026-05-22T150830Z', 24, False),
        (3, '2026-05-28T154840Z', 24, False),
        # (8, '2026-05-29T155941Z', 24, False),   # day 8

        # (1, '2026-04-25T161723Z', 40, False),   # first: COMBINE TWO SESSIONS
        # (1, '2026-04-25T163456Z', 22, False),
        # (2, '2026-05-22T150830Z', 24, False),   # middle
        # (3, '2026-05-29T155941Z', 24, False),   # last
    ],
    # Block-protocol mouse: 24-to-end AND drop the mid-session ON (cue) trials.
    'MR_M01569515': [
        (1, '2026-05-12T150308Z', 24, True),
        (2, '2026-05-13T121852Z', 24, True),
        # (3, '2026-05-14T122109Z', 24, True),
        (3, '2026-05-16T173516Z', 24, True),   # day 4

        # (1, '2026-05-12T150308Z', 24, True),    # first  (spreadsheet: exclude 3 ON)
        # (2, '2026-05-29T171621Z', 24, True),    # middle (exclude ON)
        # (3, '2026-06-21T093349Z', 24, True),    # last   (exclude ON)
    ],
    'MbR_M01569518': [
        (1, '2026-05-12T132304Z', 24, True),
        # (2, '2026-05-13T135855Z', 24, True),
        # (3, '2026-05-14T135225Z', 24, True),
        (2, '2026-05-16T155926Z', 24, True),   # day 4
        # (5, '2026-05-17T154506Z', 24, True),
        (3, '2026-05-19T110125Z', 24, True),
        # (7, '2026-05-20T102711Z', 24, True),   # day 7

        # (1, '2026-04-25T144913Z', 11, False),   # first: COMBINE THREE SESSIONS (pre-block protocol, no ON exclusion)
        # (1, '2026-04-25T150252Z', 11, False),
        # (1, '2026-04-25T151326Z', 11, False),
        # (2, '2026-05-14T135225Z', 24, True),    # middle (exclude 1 ON)
        # (3, '2026-06-09T132359Z', 24, True),    # last   (exclude ON)
    ],
}


# Human-readable labels for the first/mid/last training_day numbers (1 / 2 / 3),
# used to title the phase plots.
TRAINING_PHASE_LABELS: dict[int, str] = {1: 'first', 2: 'mid', 3: 'last'}



################################################################################
# Public API
################################################################################



#===============================================================================
# 1| training_spec (Hardcoded Table -> Tidy Per-Session Spec)
#===============================================================================
def training_spec(detail: dict[str, list[tuple[int, str, int, bool]]] = TRAINING_DETAIL) -> pd.DataFrame:
    '''
    Flatten the hardcoded ``TRAINING_DETAIL`` table into a tidy per-session spec.

    ----------
    Parameters:
        detail (dict):
            The hardcoded ``{mouseID: [(training_day, session, min_trial,
            exclude_on), ...]}`` table. Defaults to ``TRAINING_DETAIL``; pass a
            trimmed copy to scope selection / filtering to specific mice.
    Returns:
        pd.DataFrame:
            One row per (mouse, session), with columns:
              * mouseID       : e.g. ``'FL_M01569519'``.
              * session       : Bonsai session-folder name, e.g. ``'2026-05-12T150308Z'``.
              * training_day  : the spreadsheet's day number (NOT the on-disk Day<N>).
              * min_trial     : keep ``trial_index >= min_trial`` for this session.
              * exclude_on    : whether to also drop this session's ON (cue) trials.
              * combine_size  : how many sessions share this (mouse, training_day).
            Rows follow the table's order.
    '''
    records = [
        {
            'mouseID': mouse,
            'session': session,
            'training_day': training_day,
            'min_trial': min_trial,
            'exclude_on': exclude_on,
        }
        for mouse, sessions in detail.items()
        for (training_day, session, min_trial, exclude_on) in sessions
    ]
    spec = pd.DataFrame.from_records(
        records,
        columns=['mouseID', 'session', 'training_day', 'min_trial', 'exclude_on'],
    )
    # How many sessions each (mouse, day) pools, so a caller can spot combined days.
    spec['combine_size'] = spec.groupby(['mouseID', 'training_day'])['session'].transform('size')
    return spec

#===============================================================================



#===============================================================================
# 2| training_session_names (Spec -> Allowlist for include=)
#===============================================================================
def training_session_names(spec: pd.DataFrame) -> list[str]:
    '''
    Return the flat list of session names to pass as ``include=`` when loading.

    These are the leaf session-folder names ``qc_datastructure`` filters on, so the
    load is scoped to exactly the training sessions the spreadsheet lists.

    ----------
    Parameters:
        spec (pd.DataFrame):
            The output of ``training_spec``.
    Returns:
        list[str]:
            Session-folder names, de-duplicated, in spec order.
    '''
    return list(dict.fromkeys(spec['session']))

#===============================================================================



#===============================================================================
# 3| summarise_training_filter (Preview the Cut / ON-Exclusion Per Session)
#===============================================================================
def summarise_training_filter(
        trials: pd.DataFrame,
        spec: pd.DataFrame,
        *,
        session_column: str = 'session',
        mouse_column: str = 'mouseID',
        trial_index_column: str = 'trial_index',
        led_column: str = 'LED',
) -> pd.DataFrame:
    '''
    Preview, per session, what ``filter_trials`` will keep and drop.

    Purpose-built to be inspected BEFORE filtering: it shows the pre-cut count and,
    for the block-protocol mice, how many ON (cue) trials sit in the kept range and
    will therefore be dropped. Sessions loaded but absent from ``spec`` are omitted
    (``filter_trials`` drops them).

    ----------
    Parameters:
        trials (pd.DataFrame):
            The unfiltered trial table from ``qc_datastructure(...).load()``.
        spec (pd.DataFrame):
            The output of ``training_spec``.
        session_column, mouse_column, trial_index_column, led_column (str):
            Column names on ``trials``. Defaults match the Q_C trial table.
    Returns:
        pd.DataFrame:
            One row per (mouseID, training_day, session), with:
            n_total, min_trial, n_after_cut, exclude_on, n_on_dropped, n_kept.
    '''
    spec_by_session = spec.set_index([mouse_column, session_column])
    rows: list[dict] = []
    for (mouse, session), session_trials in trials.groupby([mouse_column, session_column], sort=False):
        if (mouse, session) not in spec_by_session.index:
            continue
        rule = spec_by_session.loc[(mouse, session)]
        min_trial = int(rule['min_trial'])
        exclude_on = bool(rule['exclude_on'])

        after_cut = session_trials[session_trials[trial_index_column] >= min_trial]
        n_on = int((after_cut[led_column] == 'ON').sum())
        n_on_dropped = n_on if exclude_on else 0

        rows.append({
            'mouseID': mouse,
            'training_day': int(rule['training_day']),
            'session': session,
            'n_total': len(session_trials),
            'min_trial': min_trial,
            'n_after_cut': len(after_cut),
            'exclude_on': exclude_on,
            'n_on_dropped': n_on_dropped,
            'n_kept': len(after_cut) - n_on_dropped,
        })

    return pd.DataFrame.from_records(rows).sort_values(
        ['mouseID', 'training_day', 'session']
    ).reset_index(drop=True)

#===============================================================================



#===============================================================================
# 4| filter_trials (Apply the Cut + ON-Exclusion Rules)
#===============================================================================
def filter_trials(
        trials: pd.DataFrame,
        spec: pd.DataFrame,
        *,
        session_column: str = 'session',
        mouse_column: str = 'mouseID',
        trial_index_column: str = 'trial_index',
        led_column: str = 'LED',
        report: bool = True,
) -> pd.DataFrame:
    '''
    Filter a loaded trial table down to the trials the spreadsheet says to keep.

    For every session listed in ``spec``: keep ``trial_index >= min_trial``, and -
    where the spec flags ON exclusion (the MR / MbR block protocol) - drop the
    remaining ``LED == 'ON'`` cue trials. Sessions in ``trials`` that are not in
    ``spec`` are dropped (the load may include more than the spreadsheet lists).

    The kept trials are renumbered per (mouse, training_day) as ``day_trial_index``,
    so a combined day (two or three pooled sessions) reads as one 1..N sequence
    while ``trial_index`` still holds each trial's original position in its own
    session.

    ----------
    Parameters:
        trials (pd.DataFrame):
            The unfiltered trial table from ``qc_datastructure(...).load()``.
        spec (pd.DataFrame):
            The output of ``training_spec``.
        session_column, mouse_column, trial_index_column, led_column (str):
            Column names on ``trials``. Defaults match the Q_C trial table.
        report (bool):
            When True (default), print the per-session keep / drop summary.
    Returns:
        pd.DataFrame:
            The kept trials, with the original columns plus ``training_day`` and
            ``day_trial_index``, ordered by (mouseID, training_day, session,
            trial_index).
    '''
    if report:
        summary = summarise_training_filter(
            trials, spec,
            session_column=session_column, mouse_column=mouse_column,
            trial_index_column=trial_index_column, led_column=led_column,
        )
        print(summary.to_string(index=False))

    original_columns = list(trials.columns)

    # Join each trial to its session's rule; the inner join drops any loaded
    # session that is not in the spreadsheet.
    rule_columns = [mouse_column, session_column, 'training_day', 'min_trial', 'exclude_on']
    merged = trials.merge(spec[rule_columns], on=[mouse_column, session_column], how='inner')

    # Keep from the cut onwards, and drop ON trials only where the rule asks for it.
    keep = merged[trial_index_column] >= merged['min_trial']
    keep &= ~(merged['exclude_on'] & (merged[led_column] == 'ON'))
    kept = merged[keep].copy()

    # Order the combined table, then renumber trials within each (mouse, day).
    kept = kept.sort_values(
        [mouse_column, 'training_day', session_column, trial_index_column]
    ).reset_index(drop=True)
    kept['day_trial_index'] = kept.groupby([mouse_column, 'training_day']).cumcount() + 1

    return kept[original_columns + ['training_day', 'day_trial_index']]

#===============================================================================



################################################################################

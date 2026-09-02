'''
Domain-specific parsers for Bonsai event payloads.

These are small named helpers that extract structured fields from the text
strings the lab's Bonsai workflow logs into ``ExperimentEvents``. Each is
written for one specific event shape so the cell that calls
:func:`build_trial_table_for_directory` does not need to carry a regex.

Contents
--------
- parse_poke_outcome
    Pull ``success / chosen_port / correct_port / outcome`` out of a
    ``Poke: {Success=..., ChosenPort=..., CorrectPort=...}`` event string.
'''

import re


_POKE_PAYLOAD = re.compile(
    r'Poke:\s*\{Success=(?P<success>True|False),\s*'
    r'ChosenPort=(?P<chosen>-?\d+),\s*'
    r'CorrectPort=(?P<correct>\d+)-?\}'
)


def parse_poke_outcome(event_string: str) -> dict:
    '''
    Parse a Bonsai ``Poke:`` event into structured outcome fields.

    The Bonsai workflow logs the trial-end event in the form::

        Poke: {Success=True, ChosenPort=7, CorrectPort=7-}

    This function extracts the three fields, derives a categorical
    ``outcome`` (``'success'`` / ``'fail'`` / ``'miss'``), and returns
    them as a dict suitable for use as the ``outcome_parser`` argument
    of :func:`build_trial_table`.

    Parameters
    ----------
    event_string : str
        The raw event text.

    Returns
    -------
    dict
        ``{'success': bool, 'chosen_port': int, 'correct_port': int,
        'outcome': str}`` when the string matches the Poke shape;
        an empty dict otherwise (the trial row is left without these
        fields populated, which lets you spot atypical Poke variants).
    '''
    match = _POKE_PAYLOAD.search(event_string)
    if match is None:
        return {}
    chosen = int(match['chosen'])
    correct = int(match['correct'])
    success = match['success'] == 'True'
    outcome = 'success' if success else ('miss' if chosen == -1 else 'fail')
    return {
        'success': success,
        'chosen_port': chosen,
        'correct_port': correct,
        'outcome': outcome,
    }


__all__ = ['parse_poke_outcome']

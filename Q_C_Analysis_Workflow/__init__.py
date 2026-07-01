'''Initialisation file for Q_C_Analysis_Workflow.'''

__all__ = [
    'display_table',
]


def __getattr__(name):
    '''
    Lazily expose optional convenience helpers.

    Importing ``Q_C_Analysis_Workflow.new`` should not require every utility
    helper to import successfully. This keeps the parsing notebooks usable when
    a shared/trimmed copy only needs the ``new`` workflow modules.
    '''
    if name == 'display_table':
        from .utils import display_table
        return display_table
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

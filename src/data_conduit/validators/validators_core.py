"""Shared validation helpers for data-conduit objects."""


################################################################################
# Public Validation Helpers
################################################################################

def validate_dfs_dict_input(
    dfs_dict: dict | None = None,
    base_path=None,
) -> bool:
    """
    Validate the two allowed ways of obtaining a dfs_dict for collect_dfs.

    Returns True if collect_dfs should be called.
    Returns False if an existing dfs_dict should be reused.
    """
    if dfs_dict is not None:
        if not isinstance(dfs_dict, dict):
            raise TypeError(
                f"dfs_dict must be a dict if provided, got {type(dfs_dict).__name__}."
            )
        return False

    if base_path is None:
        raise ValueError(
            "Provide either dfs_dict or base_path so collect_dfs can build one."
        )

    return True

################################################################################

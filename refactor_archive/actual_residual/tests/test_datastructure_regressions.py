"""Regression checks for datastructure selection and combination boundaries."""

from pathlib import Path

import pandas as pd
import pytest

from data_conduit.actual.datastructures import (
    DataStructure,
    SessionRef,
    StreamCatalog,
    StreamContainer,
    select_sessions,
)


def _frame(*, index_name: str | None = "Time", **columns) -> pd.DataFrame:
    """Build a one-row stream with an explicitly controlled index name."""
    values = columns or {"value": [1.0]}
    return pd.DataFrame(values, index=pd.Index([0.0], name=index_name))


def test_select_sessions_rejects_invalid_hierarchy_configuration(tmp_path: Path) -> None:
    """Depth, level names, and selector levels must describe a valid hierarchy."""
    (tmp_path / "session").mkdir()

    with pytest.raises(ValueError, match="non-negative"):
        select_sessions(tmp_path, depth=-1)
    with pytest.raises(ValueError, match="must be unique"):
        select_sessions(tmp_path, depth=1, level_names=("mouse", "mouse"))
    with pytest.raises(ValueError, match="outside depth"):
        select_sessions(tmp_path, depth=0, l0_selector="mouse")


def test_duplicate_session_ids_are_stable_across_level_filters(tmp_path: Path) -> None:
    """Duplicate basenames should always use hierarchy-relative identifiers."""
    for mouse in ("mouse_a", "mouse_b"):
        (tmp_path / mouse / "same_session").mkdir(parents=True)

    all_sessions = select_sessions(
        tmp_path,
        depth=1,
        level_names=("mouseID",),
    )
    just_b = select_sessions(
        tmp_path,
        depth=1,
        level_names=("mouseID",),
        l0_selector="mouse_b",
    )

    assert set(all_sessions) == {"mouse_a/same_session", "mouse_b/same_session"}
    assert set(just_b) == {"mouse_b/same_session"}


def test_dataframe_combination_rejects_inconsistent_index_names(tmp_path: Path) -> None:
    """Combining differently named scientific axes must not silently erase them."""
    refs = {
        "a": SessionRef(tmp_path / "a", {}, {}),
        "b": SessionRef(tmp_path / "b", {}, {}),
    }
    container = StreamContainer(
        "events",
        streams={"a": _frame(index_name="Time"), "b": _frame(index_name="timestamp")},
        sessions=refs,
    )

    with pytest.raises(ValueError, match="mixes DataFrame index names"):
        container.combine()


@pytest.mark.parametrize("collision", ["session", "mouseID"])
def test_dataframe_combination_rejects_identity_collisions(
    tmp_path: Path,
    collision: str,
) -> None:
    """Source data must not be overwritten by generated provenance columns."""
    ref = SessionRef(tmp_path / "a", {"mouseID": "mouse_a"}, {})
    container = StreamContainer(
        "events",
        streams={"a": _frame(**{collision: ["measured"]})},
        sessions={"a": ref},
    )

    with pytest.raises(ValueError, match="collides"):
        container.combine()


def test_select_invalidates_loaded_results(tmp_path: Path) -> None:
    """A fresh selection must not remain paired with data from an older load."""
    (tmp_path / "session_a").mkdir()
    catalog = StreamCatalog({"events": lambda path: _frame(value=[path.name])})
    structure = DataStructure(tmp_path, catalog)
    structure.load()

    (tmp_path / "session_b").mkdir()
    sessions = structure.select()

    assert set(sessions) == {"session_a", "session_b"}
    assert structure.containers == {}
    assert structure.data is None


def test_failed_reload_preserves_the_last_complete_state(tmp_path: Path) -> None:
    """Selection, containers, and data should commit atomically after a load."""
    (tmp_path / "session_a").mkdir()
    should_fail = False

    def read(path: Path) -> pd.DataFrame:
        if should_fail:
            raise RuntimeError("reader failed")
        return _frame(value=[path.name])

    structure = DataStructure(tmp_path, StreamCatalog({"events": read}))
    previous_data = structure.load()
    previous_sessions = structure.sessions
    previous_containers = structure.containers

    (tmp_path / "session_b").mkdir()
    should_fail = True
    with pytest.raises(RuntimeError, match="reader failed"):
        structure.load()

    assert structure.sessions is previous_sessions
    assert structure.containers is previous_containers
    assert structure.data is previous_data

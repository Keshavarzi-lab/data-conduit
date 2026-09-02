"""Tests for stream helpers and the compact ``StreamContainer`` class."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data_conduit.datastructures import streams as streams_module
from data_conduit.datastructures.selection import SessionRef
from data_conduit.datastructures.streams import StreamContainer


def _session(name: str, **levels: object) -> SessionRef:
    return SessionRef(
        path=Path(name),
        levels=dict(levels),
        metadata={"not_broadcast": name},
    )


def test_stream_container_uses_in_class_init_and_direct_function_bindings() -> None:
    """Construction stays in the class while other operations bind helpers directly."""

    class_namespace = StreamContainer.__dict__
    assert class_namespace["__init__"].__qualname__ == "StreamContainer.__init__"
    assert class_namespace["add"] is streams_module._add_stream_member

    session_ids = class_namespace["session_ids"]
    assert isinstance(session_ids, property)
    assert session_ids.fget is streams_module._get_stream_session_ids
    assert session_ids.fset is None

    assert class_namespace["__len__"] is streams_module._count_stream_members
    assert class_namespace["combine"] is streams_module._combine_stream_container


def test_stream_container_copies_and_validates_initial_members() -> None:
    """Construction should copy inputs and reject mismatched member keys."""

    dataframe = pd.DataFrame({"value": [1]})
    session = _session("s1")
    streams = {"s1": dataframe}
    sessions = {"s1": session}

    container = StreamContainer("events", streams, sessions)
    streams.clear()
    sessions.clear()

    assert container.session_ids == ["s1"]
    assert len(container) == 1

    with pytest.raises(ValueError, match="same session ids"):
        StreamContainer("events", {"s1": dataframe}, {})


def test_constructor_delegates_initial_member_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-class constructor should delegate substantive member validation."""

    dataframe = pd.DataFrame({"value": [1]})
    session = _session("s1")
    streams = {"s1": dataframe}
    sessions = {"s1": session}
    captured: dict[str, object] = {}

    def fake_validate(
        name: str,
        streams_arg: dict[str, streams_module.DataObject],
        sessions_arg: dict[str, SessionRef],
    ) -> None:
        captured.update(
            name=name,
            streams=streams_arg,
            sessions=sessions_arg,
        )

    monkeypatch.setattr(streams_module, "_validate_container_members", fake_validate)

    container = StreamContainer("events", streams, sessions)

    assert captured["name"] == "events"
    assert captured["streams"] is container.streams
    assert captured["sessions"] is container.sessions
    assert container.streams is not streams
    assert container.sessions is not sessions


def test_add_delegates_stream_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bound add implementation should delegate substantive type validation."""

    container = StreamContainer("events")
    session = _session("s1")
    dataframe = pd.DataFrame({"value": [1]})
    captured: dict[str, object] = {}

    def fake_validate(
        stream: streams_module.DataObject,
        *,
        stream_name: str,
        session_id: str,
    ) -> None:
        captured.update(
            stream=stream,
            stream_name=stream_name,
            session_id=session_id,
        )

    monkeypatch.setattr(streams_module, "_validate_stream_object", fake_validate)

    assert container.add("s1", session, dataframe) is container
    assert captured["stream"] is dataframe
    assert captured["stream_name"] == "events"
    assert captured["session_id"] == "s1"
    assert container.streams["s1"] is dataframe
    assert container.sessions["s1"] is session


def test_combine_dispatches_dataframes_and_then_attaches_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound combine implementation should orchestrate the DataFrame helpers."""

    dataframe = pd.DataFrame({"value": [1]})
    session = _session("s1")
    container = StreamContainer("events", {"s1": dataframe}, {"s1": session})
    combined = pd.DataFrame({"source": ["s1"]})
    result = combined.assign(mouseID="mouse_a")
    calls: list[str] = []
    captured: dict[str, object] = {}

    def fake_combine(
        name: str,
        session_ids: list[str],
        objects: list[streams_module.DataObject],
        *,
        session_coord: str,
    ) -> streams_module.DataObject:
        calls.append("combine_dataframes")
        captured.update(
            combine_name=name,
            combine_session_ids=session_ids,
            combine_objects=objects,
            combine_session_coord=session_coord,
        )
        return combined

    def fail_xarray_combine(*args: object, **kwargs: object) -> streams_module.DataObject:
        pytest.fail("DataFrame members were dispatched to the xarray helper.")

    def fake_attach(
        data: streams_module.DataObject,
        sessions: dict[str, SessionRef],
        *,
        dim: str,
        session_coord: str,
    ) -> streams_module.DataObject:
        calls.append("attach_levels")
        captured.update(
            attach_data=data,
            attach_sessions=sessions,
            attach_dim=dim,
            attach_session_coord=session_coord,
        )
        return result

    monkeypatch.setattr(streams_module, "_combine_dataframe_streams", fake_combine)
    monkeypatch.setattr(streams_module, "_combine_xarray_streams", fail_xarray_combine)
    monkeypatch.setattr(streams_module, "_attach_stream_levels", fake_attach)

    assert container.combine(dim="sample", session_coord="source") is result
    assert calls == ["combine_dataframes", "attach_levels"]
    assert captured["combine_name"] == "events"
    assert captured["combine_session_ids"] == ["s1"]
    combine_objects = captured["combine_objects"]
    assert isinstance(combine_objects, list)
    assert len(combine_objects) == 1
    assert combine_objects[0] is dataframe
    assert captured["combine_session_coord"] == "source"
    assert captured["attach_data"] is combined
    assert captured["attach_sessions"] is container.sessions
    assert captured["attach_dim"] == "sample"
    assert captured["attach_session_coord"] == "source"


def test_combine_dispatches_xarray_and_then_attaches_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound combine implementation should orchestrate the xarray helpers."""

    data_array = xr.DataArray([1.0], dims="sample")
    session = _session("s1")
    container = StreamContainer("position", {"s1": data_array}, {"s1": session})
    combined = xr.DataArray([1.0], dims="sample")
    calls: list[str] = []
    captured: dict[str, object] = {}

    def fail_dataframe_combine(
        *args: object,
        **kwargs: object,
    ) -> streams_module.DataObject:
        pytest.fail("Xarray members were dispatched to the DataFrame helper.")

    def fake_combine(
        name: str,
        session_ids: list[str],
        objects: list[streams_module.DataObject],
        *,
        dim: str,
        session_coord: str,
    ) -> streams_module.DataObject:
        calls.append("combine_xarray")
        captured.update(
            combine_name=name,
            combine_session_ids=session_ids,
            combine_objects=objects,
            combine_dim=dim,
            combine_session_coord=session_coord,
        )
        return combined

    def fake_attach(
        data: streams_module.DataObject,
        sessions: dict[str, SessionRef],
        *,
        dim: str,
        session_coord: str,
    ) -> streams_module.DataObject:
        calls.append("attach_levels")
        captured.update(
            attach_data=data,
            attach_sessions=sessions,
            attach_dim=dim,
            attach_session_coord=session_coord,
        )
        return data

    monkeypatch.setattr(streams_module, "_combine_dataframe_streams", fail_dataframe_combine)
    monkeypatch.setattr(streams_module, "_combine_xarray_streams", fake_combine)
    monkeypatch.setattr(streams_module, "_attach_stream_levels", fake_attach)

    assert container.combine(dim="sample", session_coord="source") is combined
    assert calls == ["combine_xarray", "attach_levels"]
    assert captured["combine_name"] == "position"
    assert captured["combine_session_ids"] == ["s1"]
    combine_objects = captured["combine_objects"]
    assert isinstance(combine_objects, list)
    assert len(combine_objects) == 1
    assert combine_objects[0] is data_array
    assert captured["combine_dim"] == "sample"
    assert captured["combine_session_coord"] == "source"
    assert captured["attach_data"] is combined
    assert captured["attach_sessions"] is container.sessions
    assert captured["attach_dim"] == "sample"
    assert captured["attach_session_coord"] == "source"


def test_combine_dataframes_tags_sessions_and_levels_without_mutating_inputs() -> None:
    """DataFrame combination should attach identity without mutating inputs."""

    first = pd.DataFrame({"speed": [1.0, 2.0]})
    second = pd.DataFrame({"speed": [3.0]})
    container = StreamContainer("speed")
    container.add("s1", _session("s1", mouseID="mouse_a"), first)
    container.add("s2", _session("s2", mouseID="mouse_b"), second)

    result = container.combine()

    assert result["speed"].tolist() == [1.0, 2.0, 3.0]
    assert result["session"].tolist() == ["s1", "s1", "s2"]
    assert result["mouseID"].tolist() == ["mouse_a", "mouse_a", "mouse_b"]
    assert "not_broadcast" not in result
    assert isinstance(result.index, pd.RangeIndex)
    assert "session" not in first.columns
    assert "session" not in second.columns


def test_combine_dataframes_preserves_a_common_named_index() -> None:
    """A consistently named DataFrame index should remain intact."""

    first = pd.DataFrame({"value": [1]}, index=pd.Index([0.1], name="Time"))
    second = pd.DataFrame({"value": [2]}, index=pd.Index([0.2], name="Time"))
    container = StreamContainer(
        "events",
        {"s1": first, "s2": second},
        {"s1": _session("s1"), "s2": _session("s2")},
    )

    result = container.combine()

    assert result.index.name == "Time"
    assert result.index.tolist() == [0.1, 0.2]


def test_combine_xarray_harmonises_coordinates_and_attaches_levels() -> None:
    """Xarray combination should harmonise coordinates and attach identity."""

    first = xr.DataArray(
        [1.0, 2.0],
        dims="Time",
        coords={"quality": ("Time", ["good", "good"])},
    )
    second = xr.DataArray([3.0], dims="Time")
    container = StreamContainer(
        "position",
        {"s1": first, "s2": second},
        {
            "s1": _session("s1", mouseID="mouse_a"),
            "s2": _session("s2", mouseID="mouse_b"),
        },
    )

    result = container.combine()

    assert result.values.tolist() == [1.0, 2.0, 3.0]
    assert result.coords["session"].values.tolist() == ["s1", "s1", "s2"]
    assert result.coords["mouseID"].values.tolist() == ["mouse_a", "mouse_a", "mouse_b"]
    assert result.coords["quality"].values[:2].tolist() == ["good", "good"]
    assert pd.isna(result.coords["quality"].values[2])
    assert "session" not in first.coords
    assert "session" not in second.coords


def test_combine_rejects_empty_or_mixed_streams() -> None:
    """Combination should reject empty and heterogeneous containers."""

    with pytest.raises(ValueError, match="no session data"):
        StreamContainer("empty").combine()

    container = StreamContainer(
        "mixed",
        {
            "s1": pd.DataFrame({"value": [1]}),
            "s2": xr.DataArray(np.array([2]), dims="Time"),
        },
        {"s1": _session("s1"), "s2": _session("s2")},
    )

    with pytest.raises(TypeError, match="mixes DataFrame and non-DataFrame"):
        container.combine()

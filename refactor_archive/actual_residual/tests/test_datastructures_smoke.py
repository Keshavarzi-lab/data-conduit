"""Small end-to-end checks for the staged datastructure pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data_conduit.actual.datastructures import (
    DataStructure,
    SessionRef,
    StreamCatalog,
    StreamContainer,
)


def test_dataframe_pipeline_combines_selected_sessions(tmp_path: Path) -> None:
    """A catalog should load and tag one DataFrame from each selected session."""
    for name in ("session_a", "session_b"):
        (tmp_path / name).mkdir()

    catalog = StreamCatalog(
        {
            "events": lambda path: pd.DataFrame(
                {"value": [len(path.name)]},
                index=pd.Index([0.0], name="Time"),
            )
        }
    )

    result = DataStructure(tmp_path, catalog).load()

    assert list(result) == ["events"]
    assert result["events"]["session"].tolist() == ["session_a", "session_b"]


def test_xarray_combination_harmonises_auxiliary_coordinates(tmp_path: Path) -> None:
    """Nested provenance coordinates may be present in only some session members."""
    first_ref = SessionRef(tmp_path / "a", {"mouse": "m1"}, {})
    second_ref = SessionRef(tmp_path / "b", {"mouse": "m2"}, {})
    first = xr.DataArray(
        [1.0, 2.0],
        dims="Time",
        coords={"Time": [0.0, 1.0], "prior_session": ("Time", ["x", "x"])},
    )
    second = xr.DataArray([3.0], dims="Time", coords={"Time": [0.0]})

    combined = StreamContainer(
        "signal",
        streams={"a": first, "b": second},
        sessions={"a": first_ref, "b": second_ref},
    ).combine()

    assert combined.sizes["Time"] == 3
    assert combined.coords["prior_session"].values[:2].tolist() == ["x", "x"]
    assert pd.isna(combined.coords["prior_session"].values[2])
    np.testing.assert_array_equal(combined.coords["mouse"], ["m1", "m1", "m2"])

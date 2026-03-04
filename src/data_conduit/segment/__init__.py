"""Segmentation helpers for time-series data."""

from data_conduit.segment.segment_core import (
	segment_boolean_series,
	slice_dataarray_windows,
	slice_event_windows,
)

__all__ = [
	"segment_boolean_series",
	"slice_event_windows",
	"slice_dataarray_windows",
]


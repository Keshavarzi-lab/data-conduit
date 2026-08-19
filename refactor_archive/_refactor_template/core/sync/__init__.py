"""Synchronisation subpackage for data-conduit."""

from data_conduit._refactor_template.core.sync.ttl import (
	TTLSyncModel,
	build_pulse_table,
	extract_ttl_segments,
	align_pulse_tables,
	fit_linear_timebase,
	convert_timebase,
	get_ttl_timebase_conversion,
	get_npx_to_bonsai_time_conversion,
	plot_ttl_pulse_trains,
	plot_stacked_pulses,
	plot_timebase_fit,
	plot_conversion_error_tools,
)

__all__ = [
	"TTLSyncModel",
	"build_pulse_table",
	"extract_ttl_segments",
	"align_pulse_tables",
	"fit_linear_timebase",
	"convert_timebase",
	"get_ttl_timebase_conversion",
	"get_npx_to_bonsai_time_conversion",
	"plot_ttl_pulse_trains",
	"plot_stacked_pulses",
	"plot_timebase_fit",
	"plot_conversion_error_tools",
]

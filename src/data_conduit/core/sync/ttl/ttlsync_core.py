'''
TTL synchronisation core utilities.
----------------------------------

Description:
	Utilities for extracting TTL pulse segments, aligning pulse tables from
	different clocks, and fitting/applying a linear conversion model between
	timebases.

	The module is intentionally split into small, composable functions:
	1) segment extraction from a raw TTL-like signal
	2) pulse-table alignment across clock domains
	3) linear model fitting and transform application
	4) convenience wrappers that bundle the end-to-end conversion flow

	
	
Contents
----------
- `build_pulse_table`: Pair rise/fall timestamps into a structured pulse table.
- `extract_ttl_segments`: Extract run-length encoded TTL segments from a sampled signal.
- `align_pulse_tables`: Align reference and target pulse tables for one-to-one fitting.
- `fit_linear_timebase`: Fit a linear model mapping target clock to reference clock.
- `convert_timebase`: Apply a linear timebase transformation to values.
- `get_ttl_timebase_conversion`: End-to-end helper to convert target pulse table into reference timebase with optional visualisation.
- `get_npx_to_bonsai_time_conversion`: Convenience wrapper for NPX->Bonsai TTL timebase conversion.
- `TTLSyncModel`: Dataclass encapsulating a linear TTL sync model with fit stats and transform method.
'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring


################################################################################
# Imports
################################################################################

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

################################################################################




################################################################################
# TTL Segment Extraction
################################################################################

def build_pulse_table(
	rise_times: np.ndarray | list[float],
	fall_times: np.ndarray | list[float],
	*,
	min_duration: float = 0.0,
) -> pd.DataFrame:
	'''
	Pair rise and fall timestamps into a pulse table.

	For each rise, the immediately following fall is found via searchsorted.
	This correctly handles recordings that start mid-pulse (where a fall arrives
	before the first rise) — those orphaned falls are skipped.

	Parameters
	----------
	rise_times : array-like
		Timestamps of rising edges (pulse starts).
	fall_times : array-like
		Timestamps of falling edges (pulse ends).
	min_duration : float
		Pulses shorter than this are dropped. Default 0.0 (keep all valid pairs).

	Returns
	-------
	pd.DataFrame
		Columns ``Start``, ``End``, ``Duration`` with a 1-based ``Pulse`` index.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	rises = np.asarray(rise_times, dtype=float)
	falls = np.asarray(fall_times, dtype=float)

	# Drop any leading orphaned falls that precede the first rise
	if len(rises) > 0 and len(falls) > 0:
		first_rise = rises[0]
		falls = falls[falls > first_rise - 1e-9]

	n = min(len(rises), len(falls))
	starts = rises[:n]
	ends = falls[:n]
	durations = ends - starts

	valid_mask = durations >= min_duration
	starts, ends, durations = starts[valid_mask], ends[valid_mask], durations[valid_mask]

	return pd.DataFrame(
		{'Start': starts, 'End': ends, 'Duration': durations},
		index=pd.RangeIndex(1, len(starts) + 1, name='Pulse'),
	)


def extract_ttl_segments(
	times: np.ndarray | list[float] | pd.Index,
	values: np.ndarray | list[float] | pd.Series,
	*,
	threshold: float = 0.5,
	active_high: bool = True,
	min_duration: float = 0.0,
	drop_inactive: bool = True,
	align_to_zero: bool = True,
) -> tuple[pd.DataFrame, float]:
	'''
	Extract run-length TTL pulse segments from a sampled waveform.

	Parameters
	----------
	times : array-like
		Monotonic timestamps corresponding to `values`.
	values : array-like
		Signal values used to infer active/inactive TTL state.
	threshold : float
		Threshold used to binarize the signal into state 0/1.
	active_high : bool
		If True, active state is `value >= threshold`.
		If False, active state is `value < threshold`.
	min_duration : float
		Minimum segment duration. Segments shorter than this are dropped.
	drop_inactive : bool
		If True, keep only active segments (`State == 1`).
	align_to_zero : bool
		If True, subtract the first kept segment start from Start/End,
		so the first segment starts at 0.

	Returns
	-------
	tuple[pd.DataFrame, float]
		(`segments`, `offset`) where `segments` has columns
		['Start', 'End', 'Duration', 'State'] and `offset` is the original
		start value used for alignment.

	Raises
	------
	ValueError
		If `times` or `values` are not 1D or do not share the same length.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	t = np.asarray(times, dtype=float)
	v = np.asarray(values, dtype=float)

	if t.ndim != 1 or v.ndim != 1:
		raise ValueError('times and values must be 1D.')
	if len(t) != len(v):
		raise ValueError('times and values must have the same length.')
	if len(t) == 0:
		return pd.DataFrame(columns=['Start', 'End', 'Duration', 'State']), 0.0

	state = (v >= threshold).astype(int)
	if not active_high:
		state = 1 - state

	# Build run boundaries
	transitions = np.flatnonzero(np.diff(state) != 0) + 1
	starts = np.r_[0, transitions]
	ends = np.r_[transitions, len(state)]

	records: list[dict[str, Any]] = []
	for s, e in zip(starts, ends, strict=True):
		start_t = t[s]
		end_t = t[e - 1]
		duration = float(end_t - start_t)
		records.append(
			{
				'Start': float(start_t),
				'End': float(end_t),
				'Duration': duration,
				'State': int(state[s]),
			}
		)

	segments = pd.DataFrame.from_records(records)

	if min_duration > 0:
		segments = segments.loc[segments['Duration'] >= min_duration].copy()

	if drop_inactive:
		segments = segments.loc[segments['State'] == 1].copy()

	if segments.empty:
		return pd.DataFrame(columns=['Start', 'End', 'Duration', 'State']), 0.0

	offset = float(segments['Start'].iloc[0])
	if align_to_zero:
		segments.loc[:, ['Start', 'End']] = segments.loc[:, ['Start', 'End']] - offset

	return segments.reset_index(drop=True), offset


################################################################################




################################################################################
# Pulse Alignment + Timebase Conversion
################################################################################

def align_pulse_tables(
	reference_df: pd.DataFrame,
	target_df: pd.DataFrame,
	*,
	normalise_start: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	'''
	Align reference and target pulse tables for one-to-one model fitting.

	The function truncates both tables to equal length (`min(len(ref), len(target))`)
	so each row index corresponds to the same pulse ordinal in both clocks.

	Parameters
	----------
	reference_df : pd.DataFrame
		Pulse table in reference clock units. Must contain `Start` and `End`.
	target_df : pd.DataFrame
		Pulse table in target clock units. Must contain `Start` and `End`.
	normalise_start : bool
		If True, subtract each table's first Start value from Start/End.

	Returns
	-------
	tuple[pd.DataFrame, pd.DataFrame]
		Aligned copies of `(reference_df, target_df)` with matching row counts.

	Raises
	------
	TypeError
		If either input is not a pandas DataFrame.
	ValueError
		If required `Start`/`End` columns are missing.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	for name, df in (('reference_df', reference_df), ('target_df', target_df)):
		if not isinstance(df, pd.DataFrame):
			raise TypeError(f'{name} must be a pandas DataFrame.')
		if not {'Start', 'End'}.issubset(df.columns):
			raise ValueError(f"{name} must contain 'Start' and 'End' columns.")

	n_keep = min(len(reference_df), len(target_df))
	ref = reference_df.iloc[:n_keep].copy().reset_index(drop=True)
	tgt = target_df.iloc[:n_keep].copy().reset_index(drop=True)

	if normalise_start and n_keep > 0:
		ref0 = float(ref['Start'].iloc[0])
		tgt0 = float(tgt['Start'].iloc[0])
		ref.loc[:, ['Start', 'End']] = ref.loc[:, ['Start', 'End']] - ref0
		tgt.loc[:, ['Start', 'End']] = tgt.loc[:, ['Start', 'End']] - tgt0

	return ref, tgt


def fit_linear_timebase(
	reference_df: pd.DataFrame,
	target_df: pd.DataFrame,
	*,
	use: str = 'start',
) -> dict[str, float]:
	'''
	Fit a linear model that maps target clock values into reference clock values.

	Model
	-----
	`reference ~= slope * target + intercept`

	Parameters
	----------
	reference_df : pd.DataFrame
		Reference-clock pulse table.
	target_df : pd.DataFrame
		Target-clock pulse table.
	use : str
		Edge used for fitting: `'start'` or `'end'` (case-insensitive).

	Returns
	-------
	dict[str, float]
		Dictionary with keys `slope`, `intercept`, and `r2`.

	Raises
	------
	ValueError
		If the aligned pulse tables are empty.

	Warns
	-----
	UserWarning
		If only one aligned pulse is available. In that case the fit assumes
		equal clock rates (``slope = 1``) and estimates offset only.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	ref, tgt = align_pulse_tables(reference_df, target_df, normalise_start=False)
	if ref.empty or tgt.empty:
		raise ValueError('Cannot fit model with empty pulse tables.')

	col = 'Start' if use.lower() == 'start' else 'End'
	x = tgt[col].to_numpy(dtype=float)
	y = ref[col].to_numpy(dtype=float)
	if len(ref) == 1:
		warnings.warn(
			'Only one aligned TTL pulse is available; assuming equal clock rates '
			'(slope = 1) and estimating offset only.',
			UserWarning,
			stacklevel=2,
		)
		return {
			'slope': 1.0,
			'intercept': float(y[0] - x[0]),
			'r2': float('nan'),
		}

	slope, intercept = np.polyfit(x, y, 1)
	y_hat = slope * x + intercept
	ss_res = float(np.sum((y - y_hat) ** 2))
	ss_tot = float(np.sum((y - np.mean(y)) ** 2))
	r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)

	return {
		'slope': float(slope),
		'intercept': float(intercept),
		'r2': float(r2),
	}


def convert_timebase(
	values: np.ndarray | pd.Series | list[float],
	*,
	slope: float,
	intercept: float,
) -> np.ndarray:
	'''
	Apply a linear timebase transform (`slope * value + intercept`).

	Parameters
	----------
	values : np.ndarray | pd.Series | list[float]
		Values to be transformed.
	slope : float
		Linear slope.
	intercept : float
		Linear intercept.

	Returns
	-------
	np.ndarray
		Transformed values.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	arr = np.asarray(values, dtype=float)
	return slope * arr + intercept


def get_ttl_timebase_conversion(
	reference_pulses: pd.DataFrame,
	target_pulses: pd.DataFrame,
	*,
	use: str = 'start',
	normalise_start: bool = False,
	visualisation: bool = False,
	error_visualisation_tools: bool = False,
	reference_label: str = 'Reference',
	target_label: str = 'Target',
	return_details: bool = False,
) -> tuple[pd.DataFrame, float] | tuple[pd.DataFrame, float, 'TTLSyncModel', dict[str, float]]:
	'''
	Convert a target pulse table into a reference clock domain.

	This helper performs:
	1) aligning pulse tables,
	2) fitting a linear target->reference model,
	3) returning converted target pulse timestamps + conversion ratio.

	Parameters
	----------
	reference_pulses : pd.DataFrame
		Reference-clock pulse table.
	target_pulses : pd.DataFrame
		Target-clock pulse table to be converted into reference time.
	use : str
		Edge column used for fitting ('start' or 'end').
	normalise_start : bool
		If True, align both pulse tables to zero at first pulse before fitting.
	visualisation : bool
		If True, render Workflow_v2-style stacked pulse visualisation.
	error_visualisation_tools : bool
		If True, render per-pulse error scatter/hist diagnostics.
	reference_label : str
		Display label for reference pulses in stacked plots.
	target_label : str
		Display label for target pulses in stacked plots.
	return_details : bool
		If True, additionally return fitted `TTLSyncModel` and fit stats.

	Returns
	-------
	tuple
		Default: (`converted_target_df`, `conversion_ratio`)
		If `return_details=True`:
		(`converted_target_df`, `conversion_ratio`, `model`, `fit_stats`)

	Notes
	-----
	`conversion_ratio` is the fitted linear slope (`target -> reference`).
	A value near 1.0 indicates similar clock rate; deviations indicate drift/scaling.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	ref_aligned, tgt_aligned = align_pulse_tables(
		reference_df=reference_pulses,
		target_df=target_pulses,
		normalise_start=normalise_start,
	)

	fit_stats = fit_linear_timebase(ref_aligned, tgt_aligned, use=use)
	model = TTLSyncModel(**fit_stats)

	use_col = 'Start' if use.lower() == 'start' else 'End'
	converted_target_df = tgt_aligned.copy()
	converted_target_df[f'{use_col}_to_ref'] = model.transform(
		converted_target_df[use_col].to_numpy(float)
	)
	converted_target_df['Start_to_ref'] = model.transform(converted_target_df['Start'].to_numpy(float))
	converted_target_df['End_to_ref'] = model.transform(converted_target_df['End'].to_numpy(float))
	converted_target_df['Duration_to_ref'] = converted_target_df['End_to_ref'] - converted_target_df['Start_to_ref']

	if visualisation or error_visualisation_tools:
		from data_conduit.core.sync.ttl.ttl_visualisation import plot_conversion_error_tools, plot_stacked_pulses

		converted_for_plots = converted_target_df.copy()
		converted_for_plots['Start'] = converted_for_plots['Start_to_ref']
		converted_for_plots['End'] = converted_for_plots['End_to_ref']
		converted_for_plots['Duration'] = converted_for_plots['Duration_to_ref']

		if visualisation:
			plot_stacked_pulses(
				reference_pulses=ref_aligned,
				target_pulses=converted_for_plots,
				reference_label=reference_label,
				target_label=target_label,
			)

		if error_visualisation_tools:
			plot_conversion_error_tools(reference_pulses=ref_aligned, converted_target_pulses=converted_for_plots)

	conversion_ratio = float(model.slope)
	if return_details:
		return converted_target_df, conversion_ratio, model, fit_stats
	return converted_target_df, conversion_ratio


def get_npx_to_bonsai_time_conversion(
	reference_pulses: pd.DataFrame,
	target_pulses: pd.DataFrame,
	*,
	use: str = 'start',
	normalise_start: bool = False,
	visualisation: bool = False,
	error_visualisation_tools: bool = False,
	return_details: bool = False,
) -> tuple[pd.DataFrame, float] | tuple[pd.DataFrame, float, 'TTLSyncModel', dict[str, float]]:
	'''
	Convert NPX pulse times into Bonsai time using TTL pulse alignment.

	This is a semantic convenience wrapper over `get_ttl_timebase_conversion`
	with labels prefilled for NPX/Bonsai workflows.
	
	Parameters
	----------
	reference_pulses : pd.DataFrame
		Bonsai-clock pulse table.
	target_pulses : pd.DataFrame
		NPX-clock pulse table to be converted into Bonsai time.
	use : str
		Edge column used for fitting ('start' or 'end').
	normalise_start : bool
		If True, align both pulse tables to zero at first pulse before fitting.
	visualisation : bool
		If True, render Workflow_v2-style stacked pulse visualisation.
	error_visualisation_tools : bool
		If True, render per-pulse error scatter/hist diagnostics.
	return_details : bool
		If True, additionally return fitted `TTLSyncModel` and fit stats.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	return get_ttl_timebase_conversion(
		reference_pulses=reference_pulses,
		target_pulses=target_pulses,
		use=use,
		normalise_start=normalise_start,
		visualisation=visualisation,
		error_visualisation_tools=error_visualisation_tools,
		reference_label='Bonsai',
		target_label='NPX (converted)',
		return_details=return_details,
	)


@dataclass(slots=True)
class TTLSyncModel:
	'''
	Simple linear TTL synchronisation model.

	Attributes
	----------
	slope : float
		Scale factor mapping target clock units to reference units.
	intercept : float
		Offset term in reference units.
	r2 : float
		Coefficient of determination from fit.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring

	slope: float
	intercept: float
	r2: float

	@classmethod
	def fit(
		cls,
		reference_df: pd.DataFrame,
		target_df: pd.DataFrame,
		*,
		use: str = 'start',
	) -> 'TTLSyncModel':
		'''
		Fit a `TTLSyncModel` directly from reference/target pulse tables.

		Parameters
		----------
		reference_df : pd.DataFrame
			Reference-clock pulse table.
		target_df : pd.DataFrame
			Target-clock pulse table.
		use : str
			Edge used for fitting: `'start'` or `'end'`.

		Returns
		-------
		TTLSyncModel
			Fitted model instance.
		'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
		stats = fit_linear_timebase(reference_df=reference_df, target_df=target_df, use=use)
		return cls(
			slope=stats['slope'],
			intercept=stats['intercept'],
			r2=stats['r2'],
		)

	def transform(self, values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
		'''
		Transform target-clock values into reference-clock values.

		Parameters
		----------
		values : array-like
			Values in target clock units.

		Returns
		-------
		np.ndarray
			Converted values in reference clock units.
		'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
		return convert_timebase(values, slope=self.slope, intercept=self.intercept)


################################################################################

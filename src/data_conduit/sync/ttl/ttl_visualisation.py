'''
TTL synchronisation visualisation utilities.
--------------------------------------------

Description:
	Plotting helpers for inspecting TTL pulse trains, linear timebase fits,
	and per-pulse conversion error diagnostics. All functions accept pulse
	tables produced by the core TTL sync functions and return matplotlib
	figure/axes objects for further customisation.

	Functions are intentionally separated from the core sync logic so that
	matplotlib is only imported when a plot is actually requested.

Contents:
---------
- `_prep_pulse_arrays`
	Internal helper. Extracts (starts, durations) arrays from a pulse
	DataFrame, filtering to active state where applicable.

- `plot_ttl_pulse_trains`
	Overlays reference and target pulse trains on a shared axis, with the
	target stream vertically offset. Useful for a quick sanity check before
	fitting.

- `plot_timebase_fit`
	Scatter plot of reference vs target pulse edge times with the fitted
	linear model overlaid. Shows goodness-of-fit at a glance.

- `plot_stacked_pulses`
	Stacked broken-bar visualisation of reference and converted target pulses
	on dual x-axes, wrapped across multiple rows. Primary diagnostic for
	verifying post-conversion alignment.

- `plot_conversion_error_tools`
	Per-pulse error scatter plots and histograms for Start, End, and Duration
	residuals (in milliseconds). Used to verify sub-millisecond alignment
	quality after conversion.
'''																														# noqa: D206


################################################################################
# Imports
################################################################################

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	import pandas as pd

from data_conduit.sync.ttl.ttlsync_core import TTLSyncModel, align_pulse_tables, fit_linear_timebase

################################################################################




################################################################################
# Internal helpers
################################################################################


def _prep_pulse_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
	'''
	Extract (starts, durations) arrays from a pulse DataFrame.

	Filters to active-state rows when a ``State`` column is present, then
	resolves Start and Duration (or End) columns case-insensitively.

	Parameters
	----------
	df : pd.DataFrame
		Pulse table. Expected columns: ``Start`` (required), and either
		``Duration`` or ``End``. An optional ``State`` column is used to
		filter inactive rows.

	Returns
	-------
	tuple[np.ndarray, np.ndarray]
		``(starts, durations)`` — both 1-D float arrays of the same length.
		Negative durations are clamped to zero.

	Raises
	------
	ValueError
		If ``Start`` / ``Start_Time`` column is missing, or if neither
		``Duration`` nor ``End`` is present.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	df = df.copy()
	if 'State' in df.columns:
		df = df.loc[df['State'].astype(float) > 0].copy()

	cols = {c.lower(): c for c in df.columns}
	s = cols.get('start') or cols.get('start_time')
	e = cols.get('end') or cols.get('end_time')
	d = cols.get('duration')
	if s is None:
		raise ValueError('Missing Start/Start_Time column.')

	starts = df[s].astype(float).to_numpy()
	if d is not None:
		durs = df[d].astype(float).to_numpy()
	elif e is not None:
		durs = (df[e] - df[s]).astype(float).to_numpy()
	else:
		raise ValueError('Need either Duration or End column.')

	return starts, np.maximum(durs, 0.0)


################################################################################




################################################################################
# Visualisation functions
################################################################################


def plot_ttl_pulse_trains(
	reference_pulses: pd.DataFrame,
	target_pulses: pd.DataFrame,
	*,
	max_pulses: int = 80,
	target_vertical_offset: float = 1.25,
	reference_color: str = 'tab:blue',
	target_color: str = 'tab:orange',
	alpha: float = 0.35,
	ax=None,
):
	'''
	Plot aligned TTL pulse trains with target stream vertically offset.

	Both streams are normalised to start at zero before plotting. Useful
	as a quick visual sanity check that the two pulse trains are structurally
	similar before fitting a conversion model.

	Parameters
	----------
	reference_pulses : pd.DataFrame
		Reference-clock pulse table (e.g. Bonsai/HARP). Must contain
		``Start`` and ``End`` columns.
	target_pulses : pd.DataFrame
		Target-clock pulse table (e.g. Neuropixels). Must contain
		``Start`` and ``End`` columns.
	max_pulses : int
		Maximum number of pulses to render per stream. Default 80.
	target_vertical_offset : float
		Vertical offset applied to the target stream so the two trains
		do not overlap. Default 1.25.
	reference_color : str
		Matplotlib colour string for reference pulses. Default ``'tab:blue'``.
	target_color : str
		Matplotlib colour string for target pulses. Default ``'tab:orange'``.
	alpha : float
		Opacity of pulse outlines. Default 0.35.
	ax : matplotlib.axes.Axes or None
		Axes to draw on. If ``None``, a new figure is created.

	Returns
	-------
	matplotlib.axes.Axes
		The axes containing the plot.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	import matplotlib.pyplot as plt

	ref_aligned, tgt_aligned = align_pulse_tables(reference_pulses, target_pulses)
	ref_rel = ref_aligned.copy()
	tgt_rel = tgt_aligned.copy()

	if not ref_rel.empty:
		ref_rel.loc[:, ['Start', 'End']] = ref_rel.loc[:, ['Start', 'End']] - ref_rel['Start'].iloc[0]
	if not tgt_rel.empty:
		tgt_rel.loc[:, ['Start', 'End']] = tgt_rel.loc[:, ['Start', 'End']] - tgt_rel['Start'].iloc[0]

	if ax is None:
		_, ax = plt.subplots(figsize=(11, 3))

	for s, e in ref_rel[['Start', 'End']].to_numpy()[:max_pulses]:
		ax.plot([s, s, e, e], [0, 1, 1, 0], color=reference_color, alpha=alpha)
	for s, e in tgt_rel[['Start', 'End']].to_numpy()[:max_pulses]:
		ax.plot(
			[s, s, e, e],
			[target_vertical_offset, target_vertical_offset + 1, target_vertical_offset + 1, target_vertical_offset],
			color=target_color,
			alpha=alpha,
		)

	ax.set_yticks([0, 1, target_vertical_offset, target_vertical_offset + 1], ['0', '1', '0', '1'])
	ax.set_xlabel('relative time (s)')
	ax.set_ylabel('TTL state')
	ax.set_title(f'Real TTL pulse trains (first {max_pulses} pulses, target offset)')
	return ax


def plot_timebase_fit(
	reference_pulses: pd.DataFrame,
	target_pulses: pd.DataFrame,
	*,
	use: str = 'start',
	model: TTLSyncModel | None = None,
	ax=None,
):
	'''
	Plot the linear target-to-reference timebase fit.

	Scatter-plots matched pulse edge times (target on x, reference on y)
	and overlays the fitted linear model. Provides a visual check of
	linearity and goodness-of-fit.

	Parameters
	----------
	reference_pulses : pd.DataFrame
		Reference-clock pulse table. Must contain ``Start`` and ``End``.
	target_pulses : pd.DataFrame
		Target-clock pulse table. Must contain ``Start`` and ``End``.
	use : str
		Edge used for fitting and plotting: ``'start'`` or ``'end'``.
		Default ``'start'``.
	model : TTLSyncModel or None
		Pre-fitted model. If ``None``, a model is fitted from the supplied
		pulse tables.
	ax : matplotlib.axes.Axes or None
		Axes to draw on. If ``None``, a new figure is created.

	Returns
	-------
	matplotlib.axes.Axes
		The axes containing the plot.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	import matplotlib.pyplot as plt

	ref_aligned, tgt_aligned = align_pulse_tables(reference_pulses, target_pulses, normalise_start=False)
	if model is None:
		model = TTLSyncModel(**fit_linear_timebase(ref_aligned, tgt_aligned, use=use))

	col = 'Start' if use.lower() == 'start' else 'End'
	x = tgt_aligned[col].to_numpy(float)
	y = ref_aligned[col].to_numpy(float)
	y_hat = model.slope * x + model.intercept

	order = np.argsort(x)
	x_sorted = x[order]
	y_hat_sorted = y_hat[order]

	if ax is None:
		_, ax = plt.subplots(figsize=(5.5, 4.5))

	ax.scatter(x, y, s=16, alpha=0.75, label=f'pulse {col.lower()}s')
	ax.plot(
		x_sorted,
		y_hat_sorted,
		lw=2,
		label=f'fit: y={model.slope:.6f}x+{model.intercept:.6f}',
	)
	ax.set_xlabel('target clock (s)')
	ax.set_ylabel('reference clock (s)')
	ax.set_title(f'TTL timebase fit ($R^2$={model.r2:.6f})')
	ax.legend(loc='best')
	return ax


def plot_stacked_pulses(
	reference_pulses: pd.DataFrame,
	target_pulses: pd.DataFrame,
	*,
	pulses_per_row: int | None = None,
	autofit_pulses_per_row: bool = True,
	reference_label: str = 'Reference',
	target_label: str = 'Target',
):
	'''
	Stacked broken-bar visualisation of reference and converted target pulses.

	Renders reference (blue) and target (red) pulses as horizontal broken
	bars on dual x-axes, wrapping across multiple rows. Both streams must
	be in the same time domain (i.e. pass converted target pulses, not raw
	target pulses).

	The target axis is a twin of the reference axis. Its background patch
	is made transparent so that only the reference axis background is visible.

	Parameters
	----------
	reference_pulses : pd.DataFrame
		Reference-clock pulse table. Must contain ``Start`` and either
		``End`` or ``Duration``.
	target_pulses : pd.DataFrame
		Converted target pulse table (same time domain as reference).
		Must contain ``Start`` and either ``End`` or ``Duration``.
	pulses_per_row : int or None
		Number of pulses to show per row. If ``None``, determined
		automatically from ``autofit_pulses_per_row``.
	autofit_pulses_per_row : bool
		If ``True`` and ``pulses_per_row`` is ``None``, selects a round
		number of pulses per row from a fixed set of candidates.
		Default ``True``.
	reference_label : str
		Display label for the reference stream. Default ``'Reference'``.
	target_label : str
		Display label for the target stream. Default ``'Target'``.

	Returns
	-------
	tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]] or tuple[None, None]
		``(fig, axes)`` — or ``(None, None)`` if there are no pulses to plot.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	import matplotlib.pyplot as plt

	ref_starts, ref_durs = _prep_pulse_arrays(reference_pulses)
	tgt_starts, tgt_durs = _prep_pulse_arrays(target_pulses)

	n = int(max(len(ref_starts), len(tgt_starts)))
	if n == 0:
		print('No pulses to plot.')
		return None, None

	if pulses_per_row is None and autofit_pulses_per_row:
		pulses_to_use = min(len(ref_starts), len(tgt_starts))
		if pulses_to_use <= 32:
			pulses_per_row = pulses_to_use
		else:
			for p in (200, 160, 120, 100, 80, 64, 50, 40):
				if pulses_to_use >= p:
					pulses_per_row = p
					break
			if pulses_per_row is None:
				pulses_per_row = pulses_to_use
	elif pulses_per_row is None:
		pulses_per_row = min(len(ref_starts), len(tgt_starts))

	n_rows = int(np.ceil(n / pulses_per_row))
	fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.2 * n_rows), constrained_layout=True)
	if n_rows == 1:
		axes = [axes]

	for r in range(n_rows):
		ax = axes[r]
		i0 = r * pulses_per_row
		i1 = min(i0 + pulses_per_row, n)

		seg_ref = list(zip(ref_starts[i0:i1], ref_durs[i0:i1], strict=True))
		ax.broken_barh(seg_ref, (0.2, 0.6), facecolors='#0463de')
		ax.set_xlabel(f'{reference_label} time')

		ax2 = ax.twiny()
		seg_tgt = list(zip(tgt_starts[i0:i1], tgt_durs[i0:i1], strict=True))
		ax2.broken_barh(seg_tgt, (1.2, 0.6), facecolors='#de0404')
		ax2.set_xlabel(f'{target_label} time')
		ax2.patch.set_visible(False)  # keep twin axis transparent so ax background shows through

		ax.set_yticks([0.5, 1.5])
		ax.set_yticklabels([reference_label, target_label])
		ax.set_ylim(0, 2)
		ax.set_title(f'Pulses {i0 + 1}\u2013{i1}')
		ax.grid(True, axis='x', alpha=0.3)
		ax.set_facecolor('whitesmoke')

	return fig, axes


def plot_conversion_error_tools(
	reference_pulses: pd.DataFrame,
	converted_target_pulses: pd.DataFrame,
):
	'''
	Per-pulse conversion error diagnostics: scatter plots and histograms.

	Computes the residual between converted target pulse times and reference
	pulse times for Start, End, and Duration. Renders a 3×2 grid of scatter
	plots (error vs pulse index) and histograms (error distribution) in
	milliseconds.

	Parameters
	----------
	reference_pulses : pd.DataFrame
		Reference-clock pulse table. Must contain ``Start``, ``End``, and
		``Duration``.
	converted_target_pulses : pd.DataFrame
		Target pulse table after conversion to the reference clock domain.
		Must contain ``Start``, ``End``, and ``Duration``.

	Returns
	-------
	tuple[matplotlib.figure.Figure, numpy.ndarray]
		``(fig, axs)`` where ``axs`` is a 3×2 array of axes.
	'''																														# noqa: D206 | Justification: Not really relevant whether indentation uses tabs or spaces in docstring
	import matplotlib.pyplot as plt

	ref_aligned, tgt_aligned = align_pulse_tables(
		reference_pulses,
		converted_target_pulses,
		normalise_start=True,
	)

	start_err_ms = (tgt_aligned['Start'].to_numpy(float) - ref_aligned['Start'].to_numpy(float)) * 1000
	end_err_ms = (tgt_aligned['End'].to_numpy(float) - ref_aligned['End'].to_numpy(float)) * 1000
	duration_err_ms = (tgt_aligned['Duration'].to_numpy(float) - ref_aligned['Duration'].to_numpy(float)) * 1000

	fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 9), sharex='col')
	idx = np.arange(len(ref_aligned))
	errors = [
		('Start', start_err_ms),
		('End', end_err_ms),
		('Duration', duration_err_ms),
	]

	for i, (label, data) in enumerate(errors):
		ax_scatter = axs[i, 0]
		ax_hist = axs[i, 1]

		ax_scatter.scatter(idx, data, s=10, alpha=0.7, label=f'{label} error (ms)', edgecolors='none')
		ax_scatter.set_ylabel('Error (ms)')
		if np.max(np.abs(data)) < 1:
			ax_scatter.set_ylim(-1, 1)
		ax_scatter.legend(loc='upper right', fontsize='small')
		ax_scatter.grid(True, alpha=0.2)

		ax_hist.hist(data, bins=30, alpha=0.7, label=f'{label} error (ms)')
		ax_hist.set_ylabel('Count')
		ax_hist.legend(loc='upper right', fontsize='small')
		ax_hist.grid(True, alpha=0.2)

	axs[-1, 0].set_xlabel('Pulse index')
	axs[-1, 1].set_xlabel('Error (ms)')
	axs[0, 0].set_title('Error vs pulse index')
	axs[0, 1].set_title('Error distribution')
	plt.tight_layout()
	return fig, axs


################################################################################

'''TTL synchronisation visualisation helpers.'''

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	import pandas as pd

from data_conduit.ttlsync.ttlsync_core import TTLSyncModel, align_pulse_tables, fit_linear_timebase


def _prep_pulse_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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
	'''Plot aligned TTL pulse trains with target stream vertically offset.'''
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
	'''Plot linear target->reference timebase fit using pulse Start/End edges.'''
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
	ax.set_xlabel('NPX clock (s)')
	ax.set_ylabel('Bonsai clock (s)')
	ax.set_title(f'Real-session TTL fit ($R^2$={model.r2:.6f})')
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
	'''Workflow_v2-style stacked pulse visualisation using broken bars on dual x-axes.'''
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

		ax.set_yticks([0.5, 1.5])
		ax.set_yticklabels([reference_label, target_label])
		ax.set_ylim(0, 2)
		ax.set_title(f'Pulses {i0 + 1}\u2013{i1}')
		ax.grid(True, axis='x', alpha=0.3)
		ax.set_facecolor('whitesmoke')
		ax2.set_facecolor('whitesmoke')

	return fig, axes


def plot_conversion_error_tools(
	reference_pulses: pd.DataFrame,
	converted_target_pulses: pd.DataFrame,
):
	'''Workflow_v2-style per-pulse error diagnostics (scatter + histogram).'''
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

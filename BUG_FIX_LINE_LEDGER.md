# Reviewed bug-fix line ledger

This ledger replaces the earlier claim that every change in commit `17b00bc`
was a confirmed bug fix. The review split that commit into 18 behavioural
changes: eight reproducible fixes, eight defensive or policy changes, and two
disputed API changes. No production call site in this repository was found
exhibiting these failures; "reproducible" below means that the pre-fix code
failed for a documented or reasonably valid input.

Line numbers refer to the reviewed files after promotion out of
`src/data_conduit/actual/` and after the conservative wrap-up.

## Reproducible fixes retained

| File | Current lines | Retained correction |
| --- | ---: | --- |
| `src/data_conduit/core/timestamps/timestamps_core.py` | 31, 100-103 | Give an xarray `DataArray` a collision-free temporary value-column name. The previous conversion failed for unnamed arrays and arrays named `Time`. |
| `src/data_conduit/core/timestamps/timestamps_core.py` | 312-326 | Preserve callable and collection selectors through the shared selector parser. The previous implementation wrapped a callable in a list and never invoked it. |
| `src/data_conduit/core/globaltimes/globaltimes_core.py` | 60-70 | Return `-1` for an empty target and avoid out-of-bounds indexing during exact matching. These were two separate reproducible crashes. |
| `src/data_conduit/core/globaltimes/globaltimes_core.py` | 140-141 | Return at most one timestamp when clock bounds are equal instead of returning the duplicate pair `[start, end]`. |
| `src/data_conduit/core/sync/ttl/ttl_visualisation.py` | 315-336 | Size a stacked-pulse plot from the longer input so one non-empty pulse table no longer leads to division by zero. |
| `src/data_conduit/core/sync/ttl/ttl_visualisation.py` | 393-402 | Return `(None, None)` for an empty conversion-error comparison instead of reaching a reduction over an empty array. The existing `No pulses to plot.` message remains unchanged. |
| `src/data_conduit/integrations/harp/harptools/harptools_core.py` | 487-505 | Combine the device-type selector with a caller `l0_selector`, avoiding the previous duplicate-keyword `TypeError`. |

## Explicitly approved policies retained or revised

These are recorded as user-approved behaviour, not relabelled as demonstrated
pipeline failures.

| File | Current lines | Approved behaviour |
| --- | ---: | --- |
| `src/data_conduit/core/globaltimes/globaltimes_core.py` | 143-144 | Reject a timestep whose sign points away from the requested clock end. |
| `src/data_conduit/core/globaltimes/globaltimes_core.py` | 222-236 | Allow negative, descending, and mixed-order stream timestamps. Non-ascending input emits `UserWarning`, is matched through a stable time-sorted view, and still returns indices into the original stream rows. |
| `src/data_conduit/core/sync/ttl/ttlsync_core.py` | 293-311 | Zero aligned TTL pulses still fail. One aligned pulse emits `UserWarning`, assumes `slope = 1`, estimates offset only, and reports `r2 = NaN`. Two or more pulses retain the linear fit. |
| `src/data_conduit/integrations/harp/harptools/__init__.py` | 25-34 | Re-registering the same HARP reader is a no-op; a different reader under `harp_bin` still raises. |

## Additional confirmed migration correction

| File | Current lines | Correction |
| --- | ---: | --- |
| `src/data_conduit/integrations/harp/datasource_presets/multidevice.py` | 142-152 | Forward `MultiDevice` selector kwargs to `collect_harp_dfs`. The public docstring promised this, but the implementation discarded them before the approved selector-combination logic could run. |
| `src/data_conduit/qc/trial_spec.py` | 7 | Keep the promoted `data_conduit.datastructures` namespace in the QC module description. This is a migration-only name update. |

## Defensive or disputed changes restored to the recovered behaviour

| File | Current lines | Restored behaviour and known limitation |
| --- | ---: | --- |
| `src/data_conduit/core/sync/ttl/ttlsync_core.py` | 293-313 | Removed the added `use` validator. As in the recovered code, case-insensitive `start` selects start edges and any other string selects end edges. |
| `src/data_conduit/core/sync/ttl/ttl_visualisation.py` | 315-336 | Removed the added positive-`pulses_per_row` validator. Invalid zero or negative values again fail downstream rather than through the added guard. |
| `src/data_conduit/integrations/harp/datasource_presets/devices.py` | 31, 173-178 | Removed defensive deep copies. A `Device` again retains the caller's configuration objects by reference. |
| `src/data_conduit/integrations/harp/datasource_presets/multidevice.py` | 24, 135-140 | Removed defensive deep copies. A `MultiDevice` again retains the caller's configuration objects by reference. |
| `src/data_conduit/datastructures/streams.py` | 696, 710-713 | Restored generated dataclass equality. Comparing distinct containers with pandas/xarray members may still have ambiguous value semantics; identity equality was not retained without an agreed API contract. |
| `src/data_conduit/datastructures/trials.py` | 478-482 | Restored the previous zero-buffer boundary rule: the preceding closing event is excluded from the next trial. Distinct later events at exactly the same timestamp are also excluded; the fully inclusive replacement was not retained because it assigned the previous close to both trials. |

## Evidence and verification provenance

- Commit `17b00bc` added no test files. The regression files added later in
  `92f9523` were authored after or alongside the changed policies and therefore
  were not independent evidence. They were parked with the unreviewed audit work.
- The original ledger's runtime claims depended on untracked workspace files
  absent from `17b00bc`, so that commit alone could not reproduce the claimed
  verification environment.
- The conservative wrap-up used in-memory checks only; it created no test files
  and changed no notebooks. The checks covered syntax/imports, both xarray
  timestamp-name failures, callable selectors, empty and exact timestamp
  matching, equal/wrong-direction clocks, all four mixed-order match modes,
  one-pulse TTL offset fitting, both empty and one-sided plot behaviour, direct
  and `MultiDevice` HARP selector forwarding, caller-owned preset configuration,
  restored trial boundaries, and idempotent reader registration.
- `git diff --check` passed for the reviewed source changes.

## Repository checkpoints

- `dev-refactor-full-snapshot` preserves the complete pre-wrap-up state at
  `92f9523`.
- `bb38f84` parks the unreviewed documentation, release metadata, and audit-test
  commits through ordinary Git reverts; no reset was used.
- Nothing described here has been pushed to a remote.

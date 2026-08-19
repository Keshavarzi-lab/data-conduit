# Bug-fix line ledger

This ledger records the confirmed bug fixes made after the recovered-code baseline commit `7a769d7` (`chore: snapshot recovered bug-fix targets`). The line numbers below refer to the post-fix files. The files originally lived below `src/data_conduit/actual/`; the table uses their current paths after promotion into `src/data_conduit/`. The final fix commit is reported in the handoff because a Git commit cannot record its own hash.

## Production changes

| File | Post-fix lines | Change |
| --- | ---: | --- |
| `src/data_conduit/core/timestamps/timestamps_core.py` | 31, 100-103 | Import selector parsing and give an xarray `DataArray` a collision-free temporary value-column name before conversion to a DataFrame. This fixes unnamed arrays and arrays whose name duplicates a coordinate, including `Time`. |
| `src/data_conduit/core/timestamps/timestamps_core.py` | 312-325 | Parse nested selectors with the shared selector parser so callable selectors remain callable, while retaining level-depth validation. |
| `src/data_conduit/core/globaltimes/globaltimes_core.py` | 58-68 | Return `-1` matches for an empty target array and avoid indexing past the end of the target during exact matching. |
| `src/data_conduit/core/globaltimes/globaltimes_core.py` | 138-143 | Handle equal clock bounds deterministically and reject a timestep that points away from the requested end time. |
| `src/data_conduit/core/globaltimes/globaltimes_core.py` | 214-216 | Reject non-monotonic stream timestamps before building an index map. |
| `src/data_conduit/core/sync/ttl/ttlsync_core.py` | 284-293 | Validate `use` and require at least two aligned pulses before fitting a linear timebase. |
| `src/data_conduit/core/sync/ttl/ttl_visualisation.py` | 319-336 | Validate `pulses_per_row`, handle either pulse table being empty, and defer the matplotlib import until a plot is actually needed. |
| `src/data_conduit/core/sync/ttl/ttl_visualisation.py` | 391-404 | Document and return `(None, None)` when no aligned conversion-error pulses are available, before importing matplotlib. |
| `src/data_conduit/integrations/harp/harptools/harptools_core.py` | 77, 487-502 | Combine a caller-provided level-zero selector with the mandatory HARP device-type selector instead of passing duplicate `l0_selector` keywords. |
| `src/data_conduit/integrations/harp/harptools/__init__.py` | 9, 25-34 | Make HARP reader registration idempotent while still rejecting a conflicting reader registered under the same name. |
| `src/data_conduit/integrations/harp/datasource_presets/devices.py` | 31, 174-178 | Defensively deep-copy mutable preset configuration before it is stored or transformed. |
| `src/data_conduit/integrations/harp/datasource_presets/multidevice.py` | 24, 136-142 | Defensively deep-copy mutable multi-device preset configuration before it is stored or transformed. |
| `src/data_conduit/datastructures/streams.py` | 696, 710-714 | Disable generated dataclass equality for `StreamContainer`; identity equality avoids ambiguous scalar comparisons of pandas/xarray members. |
| `src/data_conduit/datastructures/trials.py` | 478-480 | Use the documented inclusive trial window so distinct later events sharing the lower-bound timestamp are retained. |

## Focused verification

The following runtime reproductions passed against the edited files:

- xarray timestamp collection with unnamed and coordinate-colliding `DataArray` names
- callable nested timestamp selection
- empty and out-of-range timestamp index matching
- equal-bound and wrong-direction global clocks, plus non-monotonic stream rejection
- invalid timebase-fit mode and a one-pulse fit
- empty and unequal TTL diagnostic inputs
- HARP level-zero selector intersection and repeated reader registration
- independent mutable HARP preset configuration
- `StreamContainer` identity equality with tabular members
- retention of duplicate-time events at a trial boundary

All seven modules in `src/data_conduit/datastructures/` also imported successfully. The ten edited production files parsed successfully, and `git diff --check` reported no whitespace errors.

## Confirmed without further code changes

These previously identified issues were inspected and were already addressed in the recovered code, so this pass did not alter them:

- datastructure imports and forward annotations
- transactional `DataStructure.load()` behavior
- negative selection depth and duplicate level-name validation
- xarray coordinate harmonisation
- equivalent-order DLC keypoint schemas under the installed xarray version

## Non-functional file normalization

The manual patches also normalized existing end-of-file formatting: a surplus final blank line was removed from `globaltimes_core.py`, `ttlsync_core.py`, and `timestamps_core.py`, and the missing final newline in `devices.py` was added. These changes do not affect behavior.

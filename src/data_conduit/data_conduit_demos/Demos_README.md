# data-conduit demos

This folder contains lightweight workflow demos for the new `src` architecture.

## Planned demos

1. **Multi-device virtual arrays**
	- Load HARP data with `multisource.MultiDevice`.
	- Build virtual maps (`device`, `register`, `localID`).
	- Construct `data_arrays` and `lookup_arrays`.

2. **TTL synchronisation**
	- Extract pulse segments with `ttlsync.extract_ttl_segments`.
	- Align pulse tables and fit a linear time conversion model.
	- Convert one clock domain into another.

3. **Global timebase mapping**
	- Create canonical time vectors with `globaltimes.create_global_clock`.
	- Map stream timestamps to the global clock via `index_map_util`.

4. **Segmentation utilities**
	- Convert state traces to run-length segments.
	- Slice DataFrame/DataArray windows around event times.

## Suggested reading order

1. `Demo_workflow.ipynb`
2. Module APIs in:
	- `multisource/`
	- `ttlsync/`
	- `globaltimes/`
	- `segment/`

## Notes

- Demos are intentionally synthetic/minimal and should be adapted to your
  project-specific folder structure and register maps.
- The architecture is based on the new `src` modules rather than the legacy
  code in `Old/Refactor`.


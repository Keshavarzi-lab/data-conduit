'''
Bundle primitives salvaged from the old sessiongroups package.
--------------------------------------------------------------

Description:
    A "bundle" is just a dict of named data objects ({name: object}); this
    module carries the minimal primitives the datastructures layer relies on:

      * ``_object_to_bundle`` coerces one loaded source object (a MonoSource /
        MultiSource, a bare DataArray / Dataset / DataFrame, or a mapping of
        those) into named bundle members;
      * ``combine_bundles`` concatenates matching members across bundles along
        their existing alignment axis, in order, optionally tagging provenance.

    This is a Phase-A salvage: only the pieces used by ``datastructure_core``
    are lifted from the retired ``sessiongroups`` package. The Gen-2 bundle API
    (``map_bundle`` / ``select_bundle`` / ``split_bundle`` / ...) is not copied.
'''




################################################################################
# Imports
################################################################################

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

################################################################################


# Both xarray container types support the small slice of the xarray / combine
# API we rely on (.drop_vars / .coords / .assign_coords / concat), so every
# xarray branch below accepts either.
_XARRAY_TYPES = (xr.DataArray, xr.Dataset)

# Type alias: a bundle is just a dict of named data objects. The alias
# documents intent wherever it appears in a signature; it is not a distinct
# runtime type.
Bundle = dict[str, xr.DataArray | xr.Dataset | pd.DataFrame]




################################################################################
# Source-Object -> Bundle Members
################################################################################



#===============================================================================
# 1| Coerce One Loaded Source Object into Bundle Members
#===============================================================================
def _object_to_bundle(
        obj,
        name: str,
        *,
        prefix_data_arrays: bool,
) -> Bundle:
    '''
    Coerce one loaded source object into bundle members.

    A "source" can be several shapes; this normalises each to
    ``{name: object}`` entries. The checks are ordered most- to
    least-specific so the right branch fires before a more permissive one:

      1. an object with a ``.data_arrays`` mapping (MonoSource / MultiSource,
         e.g. SoundCard / Nosepoke) -> each array becomes a member,
         optionally prefixed with the source name;
      2. an object with a ``.df`` DataFrame (e.g. ExperimentEvents) -> one
         member;
      3. a bare DataArray / Dataset / DataFrame -> one member;
      4. a plain mapping of those -> members, optionally prefixed.

    ----------
    Parameters:
        obj (object):
            The loaded source object to coerce. Accepts any of the four
            shapes listed above; raises TypeError otherwise.
        name (str):
            The source name. Becomes the member key (single-member shapes)
            or the prefix (multi-member shapes).
        prefix_data_arrays (bool):
            If True, prefix each entry from ``.data_arrays`` or from a plain
            mapping with ``f'{name}:'`` so members coming from several
            sources don't collide on shared sub-names (e.g.
            ``'nosepoke:Activations'`` vs ``'soundcard:Activations'``).
    Returns:
        Bundle:
            A dict mapping member name to data object, ready to merge into
            the rest of the bundle.
    '''

    # 1| Source instance exposing named DataArrays (the common HARP case).
    data_arrays = getattr(obj, 'data_arrays', None)
    if isinstance(data_arrays, Mapping):
        # Prefix to avoid key collisions when several sources are merged.
        if prefix_data_arrays:
            return {f'{name}:{key}': value for key, value in data_arrays.items()}
        return dict(data_arrays)

    # 2| Source instance exposing a single DataFrame (e.g. ExperimentEvents).
    df = getattr(obj, 'df', None)
    if isinstance(df, pd.DataFrame):
        return {name: df}

    # 3| Already a bare data object.
    if isinstance(obj, xr.DataArray | xr.Dataset | pd.DataFrame):
        return {name: obj}

    # 4| A plain mapping of data objects.
    if isinstance(obj, Mapping):
        if prefix_data_arrays:
            return {f'{name}:{key}': value for key, value in obj.items()}
        return dict(obj)

    # No branch matched: caller passed an unsupported shape.
    raise TypeError(
        f"source {name!r} produced {type(obj).__name__}; expected an object with "
        f'.data_arrays/.df, a DataArray/Dataset/DataFrame, or a mapping of those.'
    )

#===============================================================================



################################################################################
# Private Helpers for combine_bundles
################################################################################



#===============================================================================
# 1| Drop a Provenance Coordinate / Column, If Present
#===============================================================================
def _drop_label(
        obj: xr.DataArray | xr.Dataset | pd.DataFrame,
        label: str,
) -> xr.DataArray | xr.Dataset | pd.DataFrame:
    '''
    Drop the provenance coordinate / column from one object, if present.

    ``errors='ignore'`` so this is a no-op when the label was never attached
    (e.g. an object that has not been through a combine).

    ----------
    Parameters:
        obj (xr.DataArray | xr.Dataset | pd.DataFrame):
            Object to strip.
        label (str):
            Name of the coordinate (xarray) or column (DataFrame) to drop.
    Returns:
        xr.DataArray | xr.Dataset | pd.DataFrame:
            A new object without the label; the input is not mutated.
    '''

    # xarray path: drop_vars handles both DataArray and Dataset.
    if isinstance(obj, _XARRAY_TYPES):
        return obj.drop_vars(label, errors='ignore')

    # DataFrame path: drop the column by name.
    return obj.drop(columns=label, errors='ignore')

#===============================================================================



#===============================================================================
# 2| Harmonise Along-Dim Coordinates Before xr.concat
#===============================================================================
def _harmonise_along_dim(
        objs: list,
        dim: str,
) -> list:
    '''
    Give every xarray object the same set of non-dim coords along ``dim``.

    ``xr.concat`` rejects a coordinate present in only some objects (unlike
    ``pd.concat``, which fills missing columns with NaN). That would make
    ``combine_bundles`` break as soon as it is nested (e.g. combining a
    previously-combined session (which carries a provenance coord) with a
    plain one (which does not). To stay composable, any non-dim coordinate
    that lives along ``dim`` in SOME objects is NaN-filled in the objects
    that lack it, so the concat succeeds and provenance from earlier
    combines survives.

    ----------
    Parameters:
        objs (list of xr.DataArray | xr.Dataset):
            The per-bundle arrays about to be concatenated along ``dim``.
        dim (str):
            The concatenation axis.
    Returns:
        list:
            The arrays, each carrying the union of along-``dim`` coords.
    '''

    # 1| First pass: collect the union of coordinates that (a) are not the
    #    dimension itself and (b) vary along ``dim``. These are the
    #    provenance-style coords that concat cares about.
    along_dim: dict[str, object] = {}
    for obj in objs:
        for name in obj.coords:
            if name != dim and dim in obj.coords[name].dims:
                along_dim.setdefault(name, obj.coords[name].dtype)

    # 2| Nothing varies along ``dim``: all objects already agree, concat as-is.
    if not along_dim:
        return objs

    # 3| Second pass: for each object, add any along-``dim`` coord it is
    #    missing, filled with NaN (object dtype, so it works for string-valued
    #    labels too).
    harmonised = []
    for obj in objs:
        missing = {name: dtype for name, dtype in along_dim.items() if name not in obj.coords}
        if missing:
            fills = {
                name: (dim, np.full(obj.sizes[dim], np.nan, dtype=object))
                for name in missing
            }
            obj = obj.assign_coords(fills)
        harmonised.append(obj)
    return harmonised

#===============================================================================



################################################################################
# Public API
################################################################################



#===============================================================================
# 1| Stack Matching Members Across Bundles
#===============================================================================
def combine_bundles(
        bundles: Sequence[Bundle],
        *,
        dim: str,
        keys: Sequence | None = None,
        label_coord: str = 'session',
        on: str = 'strict',
) -> Bundle:
    '''
    Stack matching members across bundles along their alignment axis, in order.

    Each bundle typically represents one session. For every reconciled
    member, the per-bundle objects are concatenated along the EXISTING
    alignment axis ``dim`` (e.g. ``'trial'`` for xarray, the row axis for
    DataFrames), so the combined member's length equals the sum of the
    per-bundle lengths and every member stays co-indexed. Concatenation
    follows the order the bundles are supplied (e.g. chronological),
    preserving session order.

    Provenance is attached ALONG the same axis (not as a new dimension) so
    the combined members remain mutually aligned: a coordinate for xarray
    members, a column for DataFrame members, both named ``label_coord``.
    Keeping provenance on the alignment axis (rather than adding a
    ``session`` dimension) is what lets per-group filtering reuse the very
    same ``select`` primitives.

    ----------
    Parameters:
        bundles (Sequence[Bundle]):
            Ordered bundles to combine. Must be non-empty.
        dim (str):
            Existing alignment axis to stack along. Names a dimension for
            xarray members; the row axis is used for DataFrame members.
        keys (Sequence | None):
            One provenance label per bundle (e.g. session ids or
            analysis-day labels). If given, each combined member carries a
            ``label_coord`` tag identifying the source bundle of every
            element (this is what ``split_bundle`` reads to reverse the
            combine). If None, no provenance is attached.
        label_coord (str):
            Name of the provenance coordinate / column. Default ``'session'``.
        on (str):
            Member-name reconciliation across bundles. ``'strict'`` (default)
            requires identical member names in every bundle;
            ``'intersection'`` keeps only names present in all bundles (in
            the first bundle's order).
    Returns:
        Bundle:
            A combined bundle: one stacked object per reconciled member name.
    '''

    # 1| Materialise once (the input may be a generator) and validate.
    bundles = list(bundles)
    if not bundles:
        raise ValueError('combine_bundles requires at least one bundle.')
    if keys is not None and len(keys) != len(bundles):
        raise ValueError(f'keys has length {len(keys)}, expected {len(bundles)}.')

    # 2| Reconcile which member names to combine across all bundles.
    if on == 'strict':
        # Strict mode: every bundle must expose the same members, else
        # stacking would silently drop or misalign data. Use the first
        # bundle's ordering as canonical.
        names = list(bundles[0])
        for i, b in enumerate(bundles[1:], start=1):
            if set(b) != set(names):
                raise ValueError(
                    f'bundle {i} member names {sorted(b)} differ from {sorted(names)}; '
                    f"pass on='intersection' to combine the shared members only."
                )
    elif on == 'intersection':
        # Lenient mode: keep only members present everywhere, preserving the
        # first bundle's order.
        common = set.intersection(*(set(b) for b in bundles))
        names = [name for name in bundles[0] if name in common]
    else:
        raise ValueError(f"on must be 'strict' or 'intersection', got {on!r}.")

    # 3| Stack each reconciled member across the bundles.
    combined: Bundle = {}
    for name in names:
        objs = [b[name] for b in bundles]
        first = objs[0]

        # 3a| xarray branch (DataArray or Dataset).
        if isinstance(first, _XARRAY_TYPES):
            # i)   Drop the label we are about to (re)write, so re-combining
            #      with the same name overwrites cleanly instead of conflicting.
            if keys is not None:
                objs = [_drop_label(obj, label_coord) for obj in objs]
            # ii)  Harmonise any remaining provenance coords so concat succeeds
            #      even when only some objects carry them (nested combine).
            objs = _harmonise_along_dim(objs, dim)
            # iii) Concatenate along the existing alignment axis.
            stacked = xr.concat(objs, dim=dim)
            # iv)  Tag each element with its source-bundle label, repeating
            #      the key once per element contributed by that bundle.
            if keys is not None:
                labels = np.concatenate(
                    [np.repeat(key, obj.sizes[dim]) for key, obj in zip(keys, objs, strict=True)]
                )
                stacked = stacked.assign_coords({label_coord: (dim, labels)})
            combined[name] = stacked

        # 3b| DataFrame branch.
        elif isinstance(first, pd.DataFrame):
            # pandas already NaN-fills mismatched columns, so a plain row
            # concat is enough; ignore_index gives a clean 0..N-1 row index.
            stacked = pd.concat(objs, ignore_index=True)
            # Same per-element provenance tag, as a column this time.
            if keys is not None:
                labels = np.concatenate(
                    [np.repeat(key, len(obj)) for key, obj in zip(keys, objs, strict=True)]
                )
                stacked[label_coord] = labels
            combined[name] = stacked

        else:
            raise TypeError(
                f'member {name!r} is {type(first).__name__}, not DataArray / Dataset / DataFrame.'
            )

    return combined

#===============================================================================



################################################################################

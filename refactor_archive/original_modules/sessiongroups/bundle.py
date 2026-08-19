'''
Alignment-preserving operations over a bundle of session data objects.
----------------------------------------------------------------------

Description:
    A bundle is an ordered dict mapping a friendly name to a data object,
    where each object is an xr.DataArray / xr.Dataset / pd.DataFrame:

        {'events': <DataFrame>, 'Activations': <DataArray>, 'PlaySoundFreq': <DataArray>}

    All the objects in one bundle describe the same session, so they share
    an alignment axis: a named dimension for xarray objects (e.g. 'Time' or
    'trial') and the row index for DataFrames. The primitives here exist so
    an operation can be applied across every object in one go without the
    objects drifting out of alignment:

      * map_bundle           :  broadcast a transform across members.
      * select_bundle        :  apply one shared indexer to every member along dim.
      * select_bundle_where  :  derive the indexer once from a reference, then
                               apply it to every member. This is the alignment
                               guarantee: one selection, applied identically
                               everywhere.
      * combine_bundles      :  concatenate matching members across bundles
                               (sessions) along dim, preserving the order in
                               which the bundles are supplied.
      * split_bundle         :  the inverse of combine_bundles.

    The dim argument is resolved PER CALL (a deliberate design choice): for
    an xarray object it names a dimension; for a DataFrame it refers to the
    row axis. This lets the same primitive drive both xarray-style and
    table-style alignment without the caller having to know which member is
    which type.

    Everything here is a pure function: inputs are never mutated, and a new
    bundle (a new dict holding new objects) is always returned.

Contents:
--------------------------------
- _as_positions:          Normalise a bool mask or int indexer into positions.
- _select_object:         Apply one indexer to one xarray/DataFrame object.
- _labels_along:          Read provenance labels off an object (coord/column).
- _drop_label:            Drop the provenance coord/column from one object.
- _harmonise_along_dim:   NaN-fill missing along-dim coords before xr.concat.
- map_bundle:             Broadcast a transform across every member.
- select_bundle:          Apply one shared indexer to every member.
- select_bundle_where:    Derive a shared indexer from a reference, then apply.
- combine_bundles:        Stack matching members across bundles along dim.
- split_bundle:           Inverse of combine_bundles, by provenance label.
'''





################################################################################
# Imports
################################################################################

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

################################################################################


# Constant: xarray container types we treat alike (DataArray = one labelled
# array; Dataset = several arrays sharing coordinates). Both expose the same
# alignment API we rely on (.isel / .assign_coords / .coords / concat), so
# every xarray branch below accepts either. Real source presets produce a
# mix: MultiSource/Nosepoke -> DataArray, MonoSource/SoundCard -> Dataset.
_XARRAY_TYPES = (xr.DataArray, xr.Dataset)

# Type alias: a bundle is just a dict of named data objects. The alias
# documents intent wherever it appears in a signature; it is not a distinct
# runtime type.
Bundle = dict[str, xr.DataArray | xr.Dataset | pd.DataFrame]




################################################################################
# Private Helpers
################################################################################



#===============================================================================
# 1| Normalise a Boolean / Integer Indexer to Positional Indices
#===============================================================================
def _as_positions(
        indexer,
        length: int,
) -> np.ndarray:
    '''
    Normalise a boolean mask or integer indexer into positional indices.

    Both forms are accepted so callers (and predicates) can return whichever
    is natural. Everything downstream then works in a single representation:
    integer positions suitable for ``.isel`` (xarray) and ``.iloc`` (pandas).

    ----------
    Parameters:
        indexer (array-like):
            Either a boolean mask (length must equal ``length``) or an array
            of integer positions.
        length (int):
            Size of the axis being selected, used to validate boolean masks.
    Returns:
        np.ndarray:
            1D array of integer positions.
    '''

    # Materialise as a numpy array so we can branch on dtype.
    arr = np.asarray(indexer)

    # Boolean mask path: validate it spans the whole axis (a mismatched
    # length is a silent-misalignment trap, so we fail loudly), then convert
    # True positions to integer indices.
    if arr.dtype == bool:
        if arr.shape[0] != length:
            raise ValueError(
                f'boolean indexer has length {arr.shape[0]}, expected {length}.'
            )
        return np.flatnonzero(arr)

    # Integer-position path: cast and flatten so any array-like shape works.
    return arr.astype(int).ravel()

#===============================================================================



#===============================================================================
# 2| Apply a Positional/Boolean Indexer to One Object
#===============================================================================
def _select_object(
        obj: xr.DataArray | xr.Dataset | pd.DataFrame,
        indexer,
        *,
        dim: str,
) -> xr.DataArray | xr.Dataset | pd.DataFrame:
    '''
    Apply an indexer to one object along its alignment axis.

    This is where the "per-call dim" design is realised: the same indexer is
    routed to ``.isel`` for an xarray object (along the named ``dim``) or to
    ``.iloc`` for a DataFrame (along its rows).

    ----------
    Parameters:
        obj (xr.DataArray | xr.Dataset | pd.DataFrame):
            Object to select from.
        indexer (array-like):
            Boolean mask or integer positions.
        dim (str):
            Named dimension for an xarray object. Ignored for a DataFrame,
            whose alignment axis is always the row index.
    Returns:
        xr.DataArray | xr.Dataset | pd.DataFrame:
            A new selected object (``obj`` is not mutated).
    '''

    # xarray path: select along the named dimension. We resolve the indexer
    # against that dimension's length and use positional selection (.isel)
    # so boolean masks and integer positions behave identically regardless
    # of the dimension's coordinate values (which may be non-unique after
    # a combine).
    if isinstance(obj, _XARRAY_TYPES):
        if dim not in obj.dims:
            raise KeyError(f'dim {dim!r} not found in xarray dims {tuple(obj.dims)}.')
        positions = _as_positions(indexer, obj.sizes[dim])
        return obj.isel({dim: positions})

    # DataFrame path: a table's records live on its rows, so alignment is
    # always row-wise. ``dim`` is accepted for a uniform call signature but
    # the row axis is what is selected.
    if isinstance(obj, pd.DataFrame):
        positions = _as_positions(indexer, len(obj))
        return obj.iloc[positions]

    raise TypeError(
        f'bundle members must be a DataArray / Dataset / DataFrame, got {type(obj).__name__}.'
    )

#===============================================================================



#===============================================================================
# 3| Read Provenance Labels From One Object
#===============================================================================
def _labels_along(
        obj: xr.DataArray | xr.Dataset | pd.DataFrame,
        label: str,
) -> np.ndarray:
    '''
    Read the provenance labels carried by one object (coordinate or column).

    ``combine_bundles`` writes a per-element provenance tag (which source
    bundle each element came from). This reads it back, again abstracting
    over the two member types: a coordinate on an xarray object, a column on
    a DataFrame.

    ----------
    Parameters:
        obj (xr.DataArray | xr.Dataset | pd.DataFrame):
            Object previously written by ``combine_bundles``.
        label (str):
            Name of the provenance coordinate (xarray) or column (DataFrame).
    Returns:
        np.ndarray:
            1D array of labels, one per element along the alignment axis.
    '''

    # xarray path: provenance lives as a (non-dimension) coordinate.
    if isinstance(obj, _XARRAY_TYPES):
        if label not in obj.coords:
            raise KeyError(
                f'label coord {label!r} not found in xarray coords {tuple(obj.coords)}.'
            )
        return np.asarray(obj.coords[label].values)

    # DataFrame path: provenance lives as a column.
    if isinstance(obj, pd.DataFrame):
        if label not in obj.columns:
            raise KeyError(f'label column {label!r} not found in DataFrame columns.')
        return obj[label].to_numpy()

    raise TypeError(
        f'bundle members must be a DataArray / Dataset / DataFrame, got {type(obj).__name__}.'
    )

#===============================================================================



#===============================================================================
# 4| Drop the Provenance Coordinate / Column From One Object
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
# 5| Harmonise Along-Dim Coordinates Before xr.concat
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




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| Broadcast a Transform Across Every Member
#===============================================================================
def map_bundle(
        bundle: Bundle,
        fn: Callable[[xr.DataArray | pd.DataFrame], xr.DataArray | pd.DataFrame],
        *,
        names: Sequence[str] | None = None,
) -> Bundle:
    '''
    Broadcast a transform across every member of a bundle.

    Use this for element-wise / per-object work where each member is
    transformed independently (e.g. unit conversion, renaming, smoothing).
    For SELECTION that must stay aligned across members, use
    ``select_bundle`` / ``select_bundle_where`` instead.

    ----------
    Parameters:
        bundle (Bundle):
            Bundle of named data objects.
        fn (callable):
            Transform applied to each selected member: ``fn(obj) -> obj``.
        names (Sequence[str] | None):
            If given, only these members are transformed; the rest pass
            through unchanged. If None (default), every member is
            transformed.
    Returns:
        Bundle:
            A new bundle with the same keys and order, holding the
            transformed objects.
    '''

    # 1| Decide which members to touch; default is all of them.
    targets = set(bundle) if names is None else set(names)

    # 2| Validate up front so a typo'd name fails clearly rather than
    #    silently doing nothing.
    missing = targets - set(bundle)
    if missing:
        raise KeyError(f'names not in bundle: {sorted(missing)}.')

    # 3| Rebuild the dict in original order: transform targets, pass the rest
    #    through untouched. A fresh dict keeps the call non-mutating.
    return {name: (fn(obj) if name in targets else obj) for name, obj in bundle.items()}

#===============================================================================



#===============================================================================
# 2| Apply One Shared Indexer to Every Member
#===============================================================================
def select_bundle(
        bundle: Bundle,
        indexer,
        *,
        dim: str,
        names: Sequence[str] | None = None,
) -> Bundle:
    '''
    Apply one shared indexer to every member along ``dim``, preserving alignment.

    The SAME ``indexer`` is applied to each member, so the members remain
    co-indexed after selection. This is the low-level entry point; usually
    you will compute the indexer from one reference member with
    ``select_bundle_where``.

    ----------
    Parameters:
        bundle (Bundle):
            Bundle of named data objects.
        indexer (array-like):
            Boolean mask or integer positions, interpreted against each
            member's alignment axis.
        dim (str):
            Named dimension for xarray members; the row axis is used for
            DataFrame members.
        names (Sequence[str] | None):
            If given, only these members are selected; the rest pass through
            unchanged. Selecting a subset deliberately can leave the bundle
            internally misaligned, so use ``names`` with care. If None
            (default), every member is selected.
    Returns:
        Bundle:
            A new bundle with the selection applied.
    '''

    # 1| Same target-resolution + validation pattern as map_bundle.
    targets = set(bundle) if names is None else set(names)
    missing = targets - set(bundle)
    if missing:
        raise KeyError(f'names not in bundle: {sorted(missing)}.')

    # 2| Apply the identical indexer to every targeted member. Because the
    #    indexer is shared, all members end up selecting the same positions
    #    along the alignment axis -> they stay aligned.
    return {
        name: (_select_object(obj, indexer, dim=dim) if name in targets else obj)
        for name, obj in bundle.items()
    }

#===============================================================================



#===============================================================================
# 3| Derive an Indexer From a Reference and Apply to Every Member
#===============================================================================
def select_bundle_where(
        bundle: Bundle,
        reference: str | xr.DataArray | pd.DataFrame,
        predicate: Callable[[xr.DataArray | pd.DataFrame], np.ndarray],
        *,
        dim: str,
        names: Sequence[str] | None = None,
) -> Bundle:
    '''
    Derive a shared indexer from a reference, then apply it to every member.

    This is the alignment guarantee in one call: the indexer is computed
    exactly once (from ``reference``) and applied identically to every
    selected member, so they cannot drift out of alignment. It is the same
    primitive used for both per-session filtering (before combining) and
    per-group filtering (after).

    ----------
    Parameters:
        bundle (Bundle):
            Bundle of named data objects.
        reference (str | xr.DataArray | pd.DataFrame):
            Object the indexer is derived from. A string is looked up as a
            member name within ``bundle`` (the common case: filter on the
            trial table, apply to every aligned array).
        predicate (callable):
            Receives the reference object and returns a boolean mask or
            integer positions: ``predicate(reference) -> indexer``.
        dim (str):
            Named dimension for xarray members; the row axis is used for
            DataFrame members.
        names (Sequence[str] | None):
            Members to select. If None (default), every member is selected.
    Returns:
        Bundle:
            A new bundle with the derived selection applied.
    '''

    # 1| Resolve the reference: a string names a bundle member; anything
    #    else is treated as the reference object directly.
    ref_obj = bundle[reference] if isinstance(reference, str) else reference

    # 2| Compute the indexer ONCE from the reference...
    indexer = predicate(ref_obj)

    # 3| ...then apply that single indexer to every member -> guaranteed alignment.
    return select_bundle(bundle, indexer, dim=dim, names=names)

#===============================================================================



#===============================================================================
# 4| Stack Matching Members Across Bundles
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



#===============================================================================
# 5| Split a Combined Bundle Back Into Per-Label Bundles
#===============================================================================
def split_bundle(
        bundle: Bundle,
        *,
        dim: str,
        by: str = 'session',
        reference: str | None = None,
        drop_label: bool = False,
) -> dict[object, Bundle]:
    '''
    Split a combined bundle back into per-label bundles (inverse of combine).

    Reads the provenance labels written by ``combine_bundles`` and
    partitions every member along ``dim`` by those labels. The split is
    computed once from a single reference member and applied identically to
    all members (via ``select_bundle``), so the per-label bundles stay
    mutually aligned), making combine_bundles / split_bundle a reversible
    round-trip.

    ----------
    Parameters:
        bundle (Bundle):
            Combined bundle carrying a provenance coordinate / column named
            ``by``.
        dim (str):
            Alignment axis to split along (the same ``dim`` passed to
            ``combine_bundles``).
        by (str):
            Name of the provenance coordinate / column. Default
            ``'session'``.
        reference (str | None):
            Member whose labels drive the split. If None (default), the
            first member is used. (All members carry the same labels after
            a combine, so the choice only matters if you have hand-built
            the bundle.)
        drop_label (bool):
            If True, drop the provenance coordinate / column from the split
            members.
    Returns:
        dict[object, Bundle]:
            Ordered mapping ``{label: bundle}``, label order following
            first appearance along ``dim``.
    '''

    # 1| Reject empty input loudly; ``next(iter(bundle))`` below would fail
    #    less helpfully.
    if not bundle:
        raise ValueError('split_bundle requires a non-empty bundle.')

    # 2| Read the per-element labels from one reference member. Because
    #    combine tags every member identically, this single label vector
    #    partitions the whole bundle consistently.
    ref_obj = bundle[reference] if reference is not None else bundle[next(iter(bundle))]
    labels = _labels_along(ref_obj, by)

    # 3| Walk the labels in first-appearance order (pd.unique preserves it),
    #    and for each one select the matching elements across every member.
    out: dict[object, Bundle] = {}
    for label in pd.unique(labels):
        sub = select_bundle(bundle, labels == label, dim=dim)
        if drop_label:
            # Optionally strip the provenance tag from the recovered pieces.
            sub = {name: _drop_label(obj, by) for name, obj in sub.items()}
        out[label] = sub
    return out

#===============================================================================



################################################################################

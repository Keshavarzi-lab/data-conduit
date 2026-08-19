'''
Session and SessionGroup: immutable wrappers over the bundle primitives.
------------------------------------------------------------------------

Description:
    A Session pairs a bundle of aligned data objects (see bundle.py) with a
    free-form metadata dict (mouse, analysis day, group label, parsed
    datetime, source path, etc.). A SessionGroup is an ordered collection of
    Sessions handled as one unit.

    These classes add no new alignment logic of their own: they are thin,
    ergonomic sugar over the functions in bundle.py. Their value is twofold:

    1. Immutability. Every operation returns a new object and never mutates
       the receiver, so the prior state is always recoverable simply by
       keeping a reference (e.g. the raw, unfiltered session stays valid
       after you filter it).
    2. Reversibility. The one lossy step (combining sessions) is explicitly
       reversible: SessionGroup.combine writes a provenance label that
       Session.split reads back, so combine / split round-trips.

    The two-stage per-session-then-per-group pattern reads as:
        group    = group.map_sessions(per_session_filter)   # stage 1: within session
        combined = group.combine(dim='trial')               # align across sessions
        result   = combined.select_where(...)               # stage 2: across sessions
        # result.split(dim='trial') recovers the per-session group

Contents:
--------------------------------
- Session:        (data, metadata) pair with map/select/select_where/split.
- SessionGroup:   ordered list of Sessions with map/filter/sorted/combine.
'''





################################################################################
# Imports
################################################################################

from collections.abc import Callable, Iterable, Sequence

import pandas as pd
import xarray as xr

from data_conduit.sessiongroups.bundle import (
    Bundle,
    combine_bundles,
    map_bundle,
    select_bundle,
    select_bundle_where,
    split_bundle,
)

################################################################################




################################################################################
# Session
################################################################################



#===============================================================================
# 1| Session Class
#===============================================================================
class Session:
    '''
    A bundle of aligned data objects for one session, plus metadata.

    A Session is essentially (data, metadata): ``data`` is the bundle whose
    members all share an alignment axis, and ``metadata`` is a free-form dict
    the builders populate (mouse, day, group label, parsed datetime, source
    path, ...) which downstream code is free to read or extend.

    All transform methods (map, select, select_where) return a NEW Session
    and leave the receiver untouched.

    ----------
    Parameters:
        data (Bundle):
            Mapping from a friendly name to an xr.DataArray / xr.Dataset /
            pd.DataFrame. All members share an alignment axis (see bundle.py).
        metadata (dict | None):
            Free-form per-session metadata (e.g. mouse, analysis day, group
            label, parsed datetime, source path). Copied on construction so
            later mutation of the caller's dict cannot reach in.
    '''

    #---------------------------------------------------------------------------
    # 1.1| Construction
    #---------------------------------------------------------------------------
    def __init__(
            self,
            data: Bundle,
            metadata: dict | None = None,
    ) -> None:
        '''
        Initialise a Session from a bundle and optional metadata.

        ----------
        Parameters:
            data (Bundle):
                The bundle of aligned data objects.
            metadata (dict | None):
                Optional metadata dict; copied on construction.
        Returns:
            None.
        '''

        # Copy both mappings so later mutation of the caller's dicts cannot
        # reach into this Session (and vice versa). The member objects
        # themselves are not copied; they are treated as immutable and only
        # ever replaced wholesale by the bundle primitives.
        self.data: Bundle = dict(data)
        self.metadata: dict = dict(metadata or {})


    #---------------------------------------------------------------------------
    # 1.2| Introspection (names / __getitem__ / __contains__ / __iter__ / __len__ / __repr__)
    #---------------------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        '''
        Return the member names in insertion order.

        ----------
        Returns:
            list[str]:
                The names of every member in this session's bundle.
        '''
        return list(self.data)

    def __getitem__(
            self,
            name: str,
    ) -> xr.DataArray | xr.Dataset | pd.DataFrame:
        '''
        Return the member object named ``name`` (enables ``session['trials']``).

        ----------
        Parameters:
            name (str):
                Member name to look up.
        Returns:
            xr.DataArray | xr.Dataset | pd.DataFrame:
                The member object.
        '''
        return self.data[name]

    def __contains__(
            self,
            name: str,
    ) -> bool:
        '''
        Return True if ``name`` is a member (enables ``'trials' in session``).

        ----------
        Parameters:
            name (str):
                Member name to test for membership.
        Returns:
            bool:
                True if the name is present in the bundle.
        '''
        return name in self.data

    def __iter__(self):
        '''
        Iterate over member names (enables ``for name in session``).

        Returns:
            Iterator over member-name strings, in insertion order.
        '''
        return iter(self.data)

    def __len__(self) -> int:
        '''
        Return the number of members in the bundle.

        Returns:
            int:
                Number of members.
        '''
        return len(self.data)

    def __repr__(self) -> str:
        '''
        Return a concise representation showing the label and member names.

        Returns:
            str:
                A short string of the form ``<Session 'label' members=[...]>``.
        '''
        # Prefer a session-level label, fall back to the group label, else blank.
        label = self.metadata.get('label') or self.metadata.get('group_label') or ''
        tag = f' {label!r}' if label else ''
        return f'<Session{tag} members={self.names}>'


    #---------------------------------------------------------------------------
    # 1.3| Non-mutating Derivation Helpers (_derive / with_metadata)
    #---------------------------------------------------------------------------
    def _derive(
            self,
            data: Bundle,
    ) -> 'Session':
        '''
        Return a new Session carrying ``data`` but this session's metadata.

        Every transform routes through here so the "return a new object,
        carry the metadata forward" contract lives in exactly one place.

        ----------
        Parameters:
            data (Bundle):
                The new bundle for the derived Session.
        Returns:
            Session:
                A new Session with the new data and the existing metadata.
        '''
        return Session(data, self.metadata)

    def with_metadata(
            self,
            **updates,
    ) -> 'Session':
        '''
        Return a new Session with ``metadata`` updated by ``updates``.

        The data bundle is shared unchanged; only the metadata dict is
        replaced (non-mutating, like every other method here).

        ----------
        Parameters:
            **updates:
                Arbitrary keyword arguments merged into the metadata dict.
        Returns:
            Session:
                A new Session whose metadata is ``{**self.metadata, **updates}``.
        '''
        # Merge new updates on top of existing metadata, then build a new
        # Session sharing the same bundle (which is itself never mutated).
        merged = {**self.metadata, **updates}
        return Session(self.data, merged)


    #---------------------------------------------------------------------------
    # 1.4| Operations Across All Data Objects (alignment-preserving)
    #---------------------------------------------------------------------------
    def map(
            self,
            fn: Callable[[xr.DataArray | pd.DataFrame], xr.DataArray | pd.DataFrame],
            *,
            names: Sequence[str] | None = None,
    ) -> 'Session':
        '''
        Broadcast ``fn`` across every member, returning a new Session.

        Delegates to bundle.map_bundle. Each member is transformed
        independently (no cross-member alignment is implied). Pass ``names``
        to restrict the transform to a subset of members.

        ----------
        Parameters:
            fn (callable):
                Function applied to each member object; receives the member
                and returns the transformed member.
            names (Sequence[str] | None):
                Subset of member names to transform. None (default) means
                every member.
        Returns:
            Session:
                A new Session whose members are the transformed copies.
        '''
        return self._derive(map_bundle(self.data, fn, names=names))

    def select(
            self,
            indexer,
            *,
            dim: str,
            names: Sequence[str] | None = None,
    ) -> 'Session':
        '''
        Apply one shared ``indexer`` to every member along ``dim``.

        Delegates to bundle.select_bundle. Because the same indexer hits
        every member, they stay aligned. Use when you already have the
        indexer (boolean mask or integer positions); otherwise prefer
        ``select_where``.

        ----------
        Parameters:
            indexer:
                A boolean mask or array of integer positions along ``dim``.
            dim (str):
                Alignment axis to slice (e.g. ``'trial'``).
            names (Sequence[str] | None):
                Subset of member names to slice. None (default) means every
                member.
        Returns:
            Session:
                A new Session whose members are the sliced copies.
        '''
        return self._derive(select_bundle(self.data, indexer, dim=dim, names=names))

    def select_where(
            self,
            reference: str | xr.DataArray | pd.DataFrame,
            predicate: Callable[[xr.DataArray | pd.DataFrame], object],
            *,
            dim: str,
            names: Sequence[str] | None = None,
    ) -> 'Session':
        '''
        Derive an indexer from ``reference`` and apply it to every member.

        Delegates to bundle.select_bundle_where. This is the alignment
        guarantee used for per-session and per-group filtering alike.

        ----------
        Parameters:
            reference (str | xr.DataArray | pd.DataFrame):
                Source object (or a member name to look up) from which the
                predicate derives an indexer.
            predicate (callable):
                Function receiving ``reference`` and returning a boolean mask
                or integer positions along ``dim``.
            dim (str):
                Alignment axis to slice along.
            names (Sequence[str] | None):
                Subset of member names to slice. None (default) means every
                member.
        Returns:
            Session:
                A new Session whose members are the filtered copies.
        '''
        return self._derive(
            select_bundle_where(self.data, reference, predicate, dim=dim, names=names)
        )


    #---------------------------------------------------------------------------
    # 1.5| Reversibility (split, inverse of SessionGroup.combine)
    #---------------------------------------------------------------------------
    def split(
            self,
            *,
            dim: str,
            by: str = 'session',
            drop_label: bool = False,
            label: str | None = None,
    ) -> 'SessionGroup':
        '''
        Split a combined session back into a SessionGroup (inverse of combine).

        Delegates to bundle.split_bundle to partition every member along
        ``dim`` by the provenance label ``by`` (written by ``combine``), then
        wraps each partition as a Session.

        ----------
        Parameters:
            dim (str):
                Alignment axis to split along (the ``dim`` used when
                combining).
            by (str):
                Provenance coordinate / column to split on. Default
                ``'session'`` (matches combine's default).
            drop_label (bool):
                If True, strip the provenance tag from the recovered sessions
                so they look identical to the originals before combine.
            label (str | None):
                Label for the returned group. Defaults to this session's
                ``group_label`` metadata.
        Returns:
            SessionGroup:
                One Session per distinct provenance label, in
                first-appearance order.
        '''

        # 1| Partition the bundle by the provenance label into {label: sub-bundle}.
        parts = split_bundle(self.data, dim=dim, by=by, drop_label=drop_label)

        # 2| Wrap each partition as a Session, recording the provenance label
        #    so the split result is self-describing.
        sessions = [Session(sub, {by: part_label}) for part_label, sub in parts.items()]

        # 3| Build the group; inherit the original group_label unless the
        #    caller overrides it via the ``label`` argument.
        return SessionGroup(sessions, label=label or self.metadata.get('group_label'))

#===============================================================================



################################################################################




################################################################################
# SessionGroup
################################################################################



#===============================================================================
# 1| SessionGroup Class
#===============================================================================
class SessionGroup:
    '''
    An ordered collection of Sessions handled as one group.

    The ordering matters: ``combine`` stacks sessions in this order, so a
    group is normally built in chronological order (the builders sort by
    parsed datetime). The per-session stage (``map_sessions`` /
    ``filter_sessions``) operates on each session independently; the
    per-group stage (``combine`` then ``select_where`` on the result)
    operates across them.

    ----------
    Parameters:
        sessions (iterable of Session):
            The member sessions, in the order they should be combined.
        label (str | None):
            Group label (e.g. ``'Testing_last'`` or a mouse id). Used by the
            split inverse to recover a labelled group.
    '''

    #---------------------------------------------------------------------------
    # 1.1| Construction
    #---------------------------------------------------------------------------
    def __init__(
            self,
            sessions: Iterable[Session],
            label: str | None = None,
    ) -> None:
        '''
        Initialise a SessionGroup from an ordered iterable of sessions.

        ----------
        Parameters:
            sessions (iterable of Session):
                The member sessions. Materialised to a list so the group is
                reusable even when a generator is passed in.
            label (str | None):
                Optional group label.
        Returns:
            None.
        '''
        # Materialise to a list so the group is reusable (the caller may pass
        # a generator) and indexable.
        self.sessions: list[Session] = list(sessions)
        self.label = label


    #---------------------------------------------------------------------------
    # 1.2| Introspection (__len__ / __iter__ / __getitem__ / __repr__)
    #---------------------------------------------------------------------------
    def __len__(self) -> int:
        '''
        Return the number of sessions in the group.

        Returns:
            int:
                Number of member Sessions.
        '''
        return len(self.sessions)

    def __iter__(self):
        '''
        Iterate over the member sessions, in order (enables ``for s in group``).

        Returns:
            Iterator over Session objects.
        '''
        return iter(self.sessions)

    def __getitem__(
            self,
            index: int,
    ) -> Session:
        '''
        Return the session at position ``index`` (enables ``group[0]``).

        ----------
        Parameters:
            index (int):
                Position in the group.
        Returns:
            Session:
                The session at that position.
        '''
        return self.sessions[index]

    def __repr__(self) -> str:
        '''
        Return a concise representation showing the label and session count.

        Returns:
            str:
                A short string of the form ``<SessionGroup 'label' sessions=N>``.
        '''
        tag = f' {self.label!r}' if self.label else ''
        return f'<SessionGroup{tag} sessions={len(self.sessions)}>'


    #---------------------------------------------------------------------------
    # 1.3| Per-Session Stage (map_sessions / filter_sessions / sorted)
    #---------------------------------------------------------------------------
    def map_sessions(
            self,
            fn: Callable[[Session], Session],
    ) -> 'SessionGroup':
        '''
        Apply ``fn`` to each session, returning a new group (stage 1).

        Per-session stage: ``fn`` receives one Session and returns one
        (typically ``s.select_where(...)``). Each session is handled
        independently, so per-session rules (e.g. dropping warm-up trials)
        need no awareness of the other sessions.

        ----------
        Parameters:
            fn (callable):
                Function ``Session -> Session`` applied independently to each
                session.
        Returns:
            SessionGroup:
                A new group, same label, with each session replaced by
                ``fn(session)``.
        '''
        # Apply fn independently to every session and rewrap in a new group.
        return SessionGroup([fn(session) for session in self.sessions], label=self.label)

    def filter_sessions(
            self,
            predicate: Callable[[Session], bool],
    ) -> 'SessionGroup':
        '''
        Keep whole sessions for which ``predicate(session)`` is True.

        Operates at the session level (drop an entire session), as opposed
        to ``map_sessions`` which transforms each session's contents.

        ----------
        Parameters:
            predicate (callable):
                Function ``Session -> bool`` deciding whether to keep each
                session.
        Returns:
            SessionGroup:
                A new group with non-matching sessions removed.
        '''
        # Walk the sessions; keep only those whose predicate evaluates True.
        return SessionGroup(
            [session for session in self.sessions if predicate(session)],
            label=self.label,
        )

    def sorted(
            self,
            key: Callable[[Session], object],
            *,
            reverse: bool = False,
    ) -> 'SessionGroup':
        '''
        Return a new group with sessions reordered by ``key`` (e.g. datetime).

        Useful to re-establish chronological order, since ``combine`` stacks
        in session order. ``key`` receives a Session (read e.g.
        ``s.metadata['datetime']``).

        ----------
        Parameters:
            key (callable):
                Function ``Session -> sortable`` used as the sort key.
            reverse (bool):
                If True, sort descending. Default False (ascending).
        Returns:
            SessionGroup:
                A new group with sessions in the new order.
        '''
        # Delegate to Python's stable sort with the supplied key function.
        return SessionGroup(
            sorted(self.sessions, key=key, reverse=reverse),
            label=self.label,
        )


    #---------------------------------------------------------------------------
    # 1.4| Per-Group Stage (combine, stack sessions into one Session)
    #---------------------------------------------------------------------------
    def combine(
            self,
            *,
            dim: str,
            keys: Sequence | None = None,
            label_coord: str = 'session',
            on: str = 'strict',
    ) -> Session:
        '''
        Combine sessions into one Session by stacking along ``dim``, in order.

        Delegates to bundle.combine_bundles. The result is a single Session
        whose members are aligned across all sessions; per-group filtering
        then uses the same select_where primitive, and ``Session.split``
        reverses it.

        ----------
        Parameters:
            dim (str):
                Existing alignment axis to stack along (e.g. ``'trial'``).
            keys (Sequence | None):
                One provenance label per session. If None (default), each
                session's ``metadata[label_coord]`` is used when present,
                else its ``'label'``, else its position, so the combined
                object can always be split back.
            label_coord (str):
                Name of the provenance coordinate / column written onto every
                member. Default ``'session'``.
            on (str):
                Member-name reconciliation policy across sessions; one of
                ``'strict'`` or ``'intersection'``. See ``combine_bundles``
                for the full semantics.
        Returns:
            Session:
                A combined Session, with metadata recording that it is
                combined and the per-session metadata it came from.
        '''

        # 1| Default the provenance labels from each session's metadata so the
        #    combine is reversible without the caller supplying keys.
        #    Preference order: an explicit label_coord value -> the generic
        #    'label' -> position.
        if keys is None:
            keys = [
                session.metadata.get(label_coord, session.metadata.get('label', index))
                for index, session in enumerate(self.sessions)
            ]

        # 2| Stack every session's bundle along ``dim``, tagging provenance.
        combined = combine_bundles(
            [session.data for session in self.sessions],
            dim=dim,
            keys=keys,
            label_coord=label_coord,
            on=on,
        )

        # 3| Record group-level metadata (including each session's metadata)
        #    so the combined Session remains self-describing and split-able.
        metadata = {
            'group_label': self.label,
            'combined': True,
            'dim': dim,
            'label_coord': label_coord,
            'sessions': [session.metadata for session in self.sessions],
        }
        return Session(combined, metadata)

#===============================================================================



################################################################################

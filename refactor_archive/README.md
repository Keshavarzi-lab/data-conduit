# Pre-promotion source archive

This directory is a temporary, non-package archive created while removing the
`data_conduit.actual` staging layer.

- `original_modules/` contains the superseded modules that previously lived
  directly under `src/data_conduit/`.
- `_refactor_template/` contains the planning/template tree used during the
  refactor.
- `actual_residual/` contains the empty staging initializer, audit/regression
  files, generated caches, and empty placeholder directories that were not
  promoted as runtime package code.

The source state was committed before these moves:

- `0440b16` preserves all runtime sources from `actual/`.
- `5c7b29c` preserves the template and outstanding legacy-source edits.

Do not import code from this archive. It can be reviewed for deletion only
after the promoted package, QC notebooks, documentation, and release metadata
have been checked.

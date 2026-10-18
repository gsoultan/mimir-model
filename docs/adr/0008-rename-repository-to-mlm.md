# ADR 0008: Rename the repository and code to mlm

Status: accepted. Date: 2026-10. Amends ADR 0006.

## Context

ADR 0006 named the format Mimir Model (mim) and gave the repository, Go
module, commands and Python package the same name. The repository is now
called mlm.

## Decision

The format keeps its name: Mimir Model, mim for short, with files ending
in `.mim` and the magics of ADR 0006. Only the code that implements it is
renamed: the Go module and package are `github.com/gsoultan/mlm` and
`mlm`, the commands are `mlm-run`, `mlm-import`, `mlm-inspect` and
`mlm-bench`, and the Python package is `mlm`.

## Consequences

- No file changes; every file that opened before opens now.
- Code that imports `github.com/gsoultan/mim` changes its import path to
  `github.com/gsoultan/mlm`, and Python code imports `mlm` instead of `mim`.

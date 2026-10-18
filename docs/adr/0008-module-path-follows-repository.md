# ADR 0008: The Go module path follows the repository

Status: accepted. Date: 2026-10.

## Context

ADR 0006 named the Go module `github.com/gsoultan/mim` and planned to
rename the repository to match. The repository was renamed to
`github.com/gsoultan/mimir-model` instead, next to `mimir` and
`mimir-server`. A module path that differs from its repository cannot be
fetched by `go get`, so every module that imported it needed a `replace`
directive pointing at the repository, and a `replace` in a library does
not reach the modules that depend on it.

## Decision

The Go module is `github.com/gsoultan/mimir-model`. The package keeps the
name `mim`, so code imports it as

    import mim "github.com/gsoultan/mimir-model"

and the subpackages as `github.com/gsoultan/mimir-model/interp`,
`.../ref`, `.../importer` and so on. The format, the file extension, the
magic, the commands and the Python package keep the names ADR 0006 gave
them.

## Consequences

- `go get github.com/gsoultan/mimir-model` works with no `replace`
  directive, in mimir, mimir-server and any other module.
- Code that imports `github.com/gsoultan/mim` changes its import path;
  nothing in the files changes.

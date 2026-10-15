# Contributing to mmf

## Running the tests

```
go test ./...
go test -race ./...
go test -run xxx -fuzz FuzzParse -fuzztime 60s .
```

CI runs `gofmt -l .` (must print nothing), `go vet ./...`, staticcheck,
`go test -race ./...`, 30 seconds of the fuzzer, and a check that the
conformance corpus is up to date.

The conformance corpus is generated, never edited by hand:

```
python3 -m pip install numpy mpmath
python3 -I tools/gen_conformance.py           # rewrite testdata/conformance
python3 -I tools/gen_conformance.py --check   # what CI runs
```

A change to a primitive's semantics or tolerance changes the generator and
the regenerated corpus in the same commit. If the reference kernels in
`ref` disagree with the corpus, fix whichever one disagrees with the spec;
never regenerate the corpus from the Go code. Run the same before
opening a pull request. A fuzzer crash is saved under `testdata/fuzz`;
commit the crasher together with the fix.

## Changing the spec

The documents under `spec/` are normative. The Go code is the reference
implementation of the container and must agree with `spec/container.md`
constant for constant.

- Any normative change (a new field, a new check, a new dtype, a new
  primitive, a change of wording that alters what a reader accepts) needs an
  ADR under `docs/adr/`, numbered in sequence, with Context, Decision and
  Consequences sections. Editorial changes do not.
- A change to `format.go`, `file.go`, `dtype.go` or `errors.go` and the
  matching change to `spec/container.md` or `spec/dtypes.md` go in the same
  commit. Add a case to `hostile_test.go` for every new rejection.
- A proposal for a new primitive must pass the admission rule in
  `spec/primitives.md` section 8: it is admitted only if no composite over
  the existing primitives can express it. Speed is not a reason. Operations
  that are expressible go to `spec/stdlib.md` as named composites.
- A new block dtype needs the byte layout, the bit-exact decode procedure
  in the form used in `spec/dtypes.md`, and test vectors.
- Identifiers are never reused: dtype ids, opcodes, section kinds, hash
  algorithm numbers, flag bits, error codes and composite names with their
  version suffix. A withdrawn identifier stays reserved.
- A change that breaks a file in `testdata/compat` is a major version bump,
  not a fix. See `spec/versioning.md`.

## Commit style

- One logical change per commit. Spec and code for the same change go
  together.
- Subject line in the imperative, under 72 characters, no trailing period:
  `Add q5_0 block dtype`, `Reject sections that overlap the preamble`.
- Body explains why, not what; reference the ADR when there is one.
- No merge commits on pull request branches; rebase on main.

## Style

Spec prose uses RFC 2119 key words, plain sentences, tables for layouts and
byte-offset tables for binary structures. Go code is `gofmt`-formatted and
passes `go vet` and staticcheck with no exceptions.

## License

Contributions are accepted under the Apache License 2.0 (see `LICENSE`).
By submitting a change you agree that it is licensed under those terms.

"""The training section of spec/training.md."""

import struct

from .errors import MIMError

TABLES = 7
HEADER_SIZE = 16 + 16 * TABLES
ROW_SIZE = (1, 8, 16, 16, 16, 8, 8)
POOL, STRINGS, COUNTERS, HYPER, PARAMS, STATES, EXTRAS = range(TABLES)
MAX_ROWS = 1 << 28
MAX_NAME_LENGTH = 4096


def _bad(detail):
    return MIMError("bad-training", detail)


def decode_training(b):
    """Decodes and validates a training section on its own. Returns a dict
    with "optimizer", "counters" (name, value), "hyperparameters" (name,
    group, value), "parameters" (tensor, group, states as (slot, tensor))
    and "extras" (name, tensor)."""
    b = memoryview(b)
    if len(b) < HEADER_SIZE or bytes(b[:4]) != b"MIMT":
        raise _bad("missing header")
    major, _minor, opt, reserved = struct.unpack_from("<HHII", b, 4)
    if major != 0:
        raise _bad(f"format major {major}, want 0")
    if reserved:
        raise _bad("reserved header bytes are not zero")
    tables = []
    for i in range(TABLES):
        off, n = struct.unpack_from("<QQ", b, 16 + 16 * i)
        size = n * ROW_SIZE[i]
        if off % 8 or n > MAX_ROWS or off > len(b) or size > len(b) - off:
            raise _bad(f"table {i} (offset {off}, {n} rows) outside the section or misaligned")
        tables.append(b[off : off + size])
    pool = tables[POOL]
    strs = []
    for off, n in struct.iter_unpack("<II", tables[STRINGS]):
        if off > len(pool) or n > len(pool) - off or n > MAX_NAME_LENGTH:
            raise _bad(f"string {len(strs)} is outside the pool or too long")
        try:
            strs.append(bytes(pool[off : off + n]).decode("utf-8"))
        except UnicodeDecodeError:
            raise _bad(f"string {len(strs)} is not UTF-8") from None
    if not strs or strs[0] != "":
        raise _bad("string 0 is not the empty string")

    def name(i, what, empty=False):
        if i >= len(strs):
            raise _bad(f"{what}: string {i} of {len(strs)}")
        if strs[i] == "" and not empty:
            raise _bad(f"{what}: empty name")
        return strs[i]

    out = {"optimizer": name(opt, "optimizer", empty=True), "counters": [], "hyperparameters": [],
           "parameters": [], "extras": []}
    seen = set()
    for n, reserved, value in struct.iter_unpack("<IIQ", tables[COUNTERS]):
        c = name(n, "counter")
        if reserved:
            raise _bad(f"counter {c!r}: reserved bytes are not zero")
        if c in seen:
            raise _bad(f"counter {c!r} appears twice")
        seen.add(c)
        out["counters"].append((c, value))
    seen = set()
    for n, group, value in struct.iter_unpack("<IId", tables[HYPER]):
        h = name(n, "hyperparameter")
        if (h, group) in seen:
            raise _bad(f"hyperparameter {h!r} of group {group} appears twice")
        seen.add((h, group))
        out["hyperparameters"].append((h, group, value))
    tensors = set()

    def tensor(i, what):
        t = name(i, what)
        if t in tensors:
            raise _bad(f"{what}: tensor {t!r} is named twice")
        tensors.add(t)
        return t

    states = list(struct.iter_unpack("<II", tables[STATES]))
    nxt = 0
    for n, group, first, count in struct.iter_unpack("<IIII", tables[PARAMS]):
        p = tensor(n, "parameter")
        if first != nxt or count > len(states) - first:
            raise _bad(f"parameter {p!r}: states {first} to {first + count} do not follow the previous parameter's")
        nxt += count
        slots, st = set(), []
        for slot, t in states[first : first + count]:
            s = name(slot, "state slot")
            if s in slots:
                raise _bad(f"parameter {p!r}: slot {s!r} appears twice")
            slots.add(s)
            st.append((s, tensor(t, "state")))
        out["parameters"].append((p, group, st))
    if nxt != len(states):
        raise _bad(f"{len(states) - nxt} states belong to no parameter")
    seen = set()
    for n, t in struct.iter_unpack("<II", tables[EXTRAS]):
        x = name(n, "extra")
        if x in seen:
            raise _bad(f"extra {x!r} appears twice")
        seen.add(x)
        out["extras"].append((x, tensor(t, "extra")))
    return out

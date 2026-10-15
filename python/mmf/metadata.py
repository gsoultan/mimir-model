"""The metadata section of spec/metadata.md."""

import struct

from .errors import MMFError

TYPES = {1: "int", 2: "uint", 3: "float", 4: "bool", 5: "string", 6: "bytes", 7: "ints", 8: "floats", 9: "strings"}
MAX_COUNT = 1 << 24
MAX_KEY_LENGTH = 1024


def decode_metadata(b):
    """Decodes a metadata section into a dict of key to (type name, value),
    in key order. Values are int, float, bool, str, bytes or lists."""
    b = memoryview(b)
    if len(b) < 16 or bytes(b[:4]) != b"MMFM":
        raise MMFError("bad-metadata", "missing header")
    n, pool_len = struct.unpack_from("<IQ", b, 4)
    if n > MAX_COUNT or 16 + 32 * n + pool_len != len(b):
        raise MMFError("bad-metadata", f"section length does not match {n} entries and {pool_len} pool bytes")
    pool = b[16 + 32 * n :]

    def in_pool(off, ln):
        return off <= len(pool) and ln <= len(pool) - off

    def string(off, ln):
        if not in_pool(off, ln):
            raise MMFError("bad-metadata", "string outside pool")
        try:
            return bytes(pool[off : off + ln]).decode("utf-8")
        except UnicodeDecodeError:
            raise MMFError("bad-metadata", "invalid UTF-8") from None

    out = {}
    prev = None
    for i in range(n):
        koff, kln, typ = struct.unpack_from("<IIB", b, 16 + 32 * i)
        e = 16 + 32 * i
        key = string(koff, kln)
        kb = key.encode("utf-8")
        if not kb or len(kb) > MAX_KEY_LENGTH or (prev is not None and kb <= prev):
            raise MMFError("bad-metadata", f"key {key!r} out of order or invalid")
        prev = kb
        if any(b[e + 9 : e + 12]) or any(b[e + 24 : e + 32]):
            raise MMFError("reserved-non-zero", f"metadata entry {i}")
        count, payload = struct.unpack_from("<IQ", b, e + 12)
        name = TYPES.get(typ)
        if name is None:
            raise MMFError("bad-metadata", f"key {key!r}: type {typ}")
        if name in ("int", "uint", "float", "bool") and count:
            raise MMFError("bad-metadata", f"key {key!r}: scalar with count")
        if name == "int":
            v = struct.unpack("<q", struct.pack("<Q", payload))[0]
        elif name == "uint":
            v = payload
        elif name == "float":
            v = struct.unpack("<d", struct.pack("<Q", payload))[0]
        elif name == "bool":
            if payload > 1:
                raise MMFError("bad-metadata", f"key {key!r}: bool value {payload}")
            v = payload == 1
        elif name == "string":
            v = string(payload, count)
        elif name == "bytes":
            if not in_pool(payload, count):
                raise MMFError("bad-metadata", f"key {key!r}: bytes outside pool")
            v = bytes(pool[payload : payload + count])
        elif name in ("ints", "floats"):
            if payload % 8 or not in_pool(payload, 8 * count):
                raise MMFError("bad-metadata", f"key {key!r}: array outside pool or misaligned")
            v = list(struct.unpack_from(f"<{count}{'q' if name == 'ints' else 'd'}", pool, payload))
        else:
            if not in_pool(payload, 8 * count):
                raise MMFError("bad-metadata", f"key {key!r}: string table outside pool")
            v = [string(*struct.unpack_from("<II", pool, payload + 8 * j)) for j in range(count)]
        out[key] = (name, v)
    return out

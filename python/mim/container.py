"""The container of spec/container.md: opening, validating and reading an
mim file. Checks run in the order of the spec, so a file with several
faults is reported with the same code as by any other conforming reader."""

import hashlib
import mmap
import struct
from collections import namedtuple

from . import dtypes as D
from .errors import MIMError

MAGIC = b"MIM\x00"
# Files written before the format was renamed from mmf start with this.
LEGACY_MAGIC = b"MMF\x00"
BYTE_ORDER_MARK = 0x0A0B0C0D
PREAMBLE_SIZE = 64
INDEX_HEADER_SIZE = 16
SECTION_ENTRY_SIZE = 64
TENSOR_ENTRY_SIZE = 96
MAX_INDEX_LENGTH = 256 << 20
MAX_RANK = 8
MAX_NAME_LENGTH = 4096
MIN_DATA_ALIGNMENT = 4 << 10
MAX_DATA_ALIGNMENT = 16 << 20
TENSOR_ALIGNMENT = 64
LARGE_TENSOR_ALIGNMENT = 4 << 10
LARGE_TENSOR_THRESHOLD = 1 << 20
SECTION_ALIGNMENT = 8
FLAG_REQUIRED = 1

WEIGHTS, METADATA, GRAPH, REPACK_CACHE, TRAINING = 1, 2, 3, 4, 5

Preamble = namedtuple("Preamble", "major minor data_alignment index_offset index_length data_offset data_length file_length")
Section = namedtuple("Section", "kind flags offset length hash_alg hash")
TensorInfo = namedtuple("TensorInfo", "name dtype shape offset length")


def _in_bounds(off, n, size):
    return off <= size and n <= size - off


def _tensor_alignment(length):
    return LARGE_TENSOR_ALIGNMENT if length >= LARGE_TENSOR_THRESHOLD else TENSOR_ALIGNMENT


def _align_up(n, a):
    return (n + a - 1) // a * a


def _pow2(n):
    return n != 0 and n & (n - 1) == 0


class File:
    """A validated, read-only view of a mim file. `data` is any buffer:
    bytes, or a memory map from open()."""

    def __init__(self, data):
        self._data = memoryview(data).cast("B")
        self._mm = None
        self._parse_preamble()
        self._parse_index()
        self._check_tensors()

    @classmethod
    def open(cls, path):
        """Maps the file read-only; nothing is copied."""
        with open(path, "rb") as fh:
            try:
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            except ValueError:  # an empty file cannot be mapped
                return cls(b"")
        f = cls(mm)
        f._mm = mm
        return f

    def close(self):
        """Unmaps the file. Tensors that still view its bytes keep the
        mapping alive until they are gone."""
        if self._mm is not None:
            try:
                self._data.release()
                self._mm.close()
            except BufferError:
                pass
            self._mm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # parsing -------------------------------------------------------------

    def _parse_preamble(self):
        b = self._data
        if len(b) < PREAMBLE_SIZE:
            raise MIMError("truncated", f"file is {len(b)} bytes, preamble needs {PREAMBLE_SIZE}")
        if bytes(b[0:4]) not in (MAGIC, LEGACY_MAGIC):
            raise MIMError("bad-magic", f"got {bytes(b[0:4])!r}")
        major, minor, bom, align, ioff, ilen, doff, dlen, flen, reserved = struct.unpack_from("<HHIIQQQQQQ", b, 4)
        if bom != BYTE_ORDER_MARK:
            if bom == 0x0D0C0B0A:
                raise MIMError("big-endian", "big-endian files are not supported")
            raise MIMError("bad-byte-order-mark", f"got {bom:#010x}")
        if major != 0:
            raise MIMError("unsupported-version", f"format {major}.{minor}, reader supports 0.x")
        if reserved != 0:
            raise MIMError("reserved-non-zero", "preamble bytes 56..63")
        if not _pow2(align) or align < MIN_DATA_ALIGNMENT or align > MAX_DATA_ALIGNMENT:
            raise MIMError("bad-alignment", f"data alignment {align}")
        if flen != len(b):
            if flen > len(b):
                raise MIMError("truncated", f"preamble says {flen} bytes, have {len(b)}")
            raise MIMError("bad-length", f"preamble says {flen} bytes, have {len(b)}")
        if ilen > MAX_INDEX_LENGTH:
            raise MIMError("index-too-large", f"index is {ilen} bytes")
        if ioff < PREAMBLE_SIZE or ioff % SECTION_ALIGNMENT:
            raise MIMError("misaligned", f"index offset {ioff}")
        if not _in_bounds(ioff, ilen, flen):
            raise MIMError("out-of-bounds", "index outside file")
        if doff % align:
            raise MIMError("misaligned", f"data offset {doff} not a multiple of {align}")
        if not _in_bounds(doff, dlen, flen):
            raise MIMError("out-of-bounds", "weights outside file")
        self.preamble = Preamble(major, minor, align, ioff, ilen, doff, dlen, flen)

    def _parse_index(self):
        p = self.preamble
        idx = self._data[p.index_offset : p.index_offset + p.index_length]
        if len(idx) < INDEX_HEADER_SIZE:
            raise MIMError("truncated", "index header")
        ns, nt, pool = struct.unpack_from("<IIQ", idx, 0)
        want = INDEX_HEADER_SIZE + ns * SECTION_ENTRY_SIZE + nt * TENSOR_ENTRY_SIZE
        if pool > MAX_INDEX_LENGTH or want + pool != len(idx):
            raise MIMError("bad-length", f"index length {len(idx)} does not match {ns} sections, {nt} tensors, {pool} pool bytes")
        self._tensors = idx[INDEX_HEADER_SIZE + ns * SECTION_ENTRY_SIZE : want]
        self._pool = idx[want:]
        self._ntensors = nt
        regions = [(0, PREAMBLE_SIZE), (p.index_offset, p.index_length)]
        seen = set()
        self.sections = []
        for i in range(ns):
            e = idx[INDEX_HEADER_SIZE + i * SECTION_ENTRY_SIZE : INDEX_HEADER_SIZE + (i + 1) * SECTION_ENTRY_SIZE]
            kind, flags, off, ln, alg = struct.unpack_from("<IIQQB", e, 0)
            h = bytes(e[32:64])
            if any(e[25:32]):
                raise MIMError("reserved-non-zero", f"section {i} reserved bytes")
            if flags & ~FLAG_REQUIRED:
                raise MIMError("reserved-non-zero", f"section {i} flags {flags:#x}")
            if alg == 0:
                if any(h):
                    raise MIMError("reserved-non-zero", f"section {i} has a hash but no algorithm")
            elif alg != 1:
                raise MIMError("bad-hash-algorithm", f"section {i} algorithm {alg}")
            if not 1 <= kind <= 5:
                if flags & FLAG_REQUIRED:
                    raise MIMError("unknown-required-section", f"section kind {kind}")
            else:
                if kind in seen:
                    raise MIMError("duplicate", f"two sections of kind {kind}")
                seen.add(kind)
            if not _in_bounds(off, ln, p.file_length):
                raise MIMError("out-of-bounds", f"section {i}")
            if kind == WEIGHTS:
                if off != p.data_offset or ln != p.data_length:
                    raise MIMError("bad-length", "weights section disagrees with preamble")
            elif off % SECTION_ALIGNMENT:
                raise MIMError("misaligned", f"section {i} offset {off}")
            self.sections.append(Section(kind, flags, off, ln, alg, h))
            regions.append((off, ln))
        if WEIGHTS not in seen:
            raise MIMError("missing-section", "no weights section")
        regions.sort()
        for (a, an), (b, _) in zip(regions, regions[1:]):
            if b < a + an:
                raise MIMError("overlap", "two regions overlap")

    def _row(self, i):
        return self._tensors[i * TENSOR_ENTRY_SIZE : (i + 1) * TENSOR_ENTRY_SIZE]

    def _name(self, r):
        off, n = struct.unpack_from("<II", r, 0)
        if not _in_bounds(off, n, len(self._pool)):
            raise MIMError("out-of-bounds", "tensor name outside string pool")
        return bytes(self._pool[off : off + n])

    def _decode_row(self, r, name):
        dt_id, rank = struct.unpack_from("<IB", r, 8)
        if any(r[13:16]):
            raise MIMError("reserved-non-zero", f"tensor {name!r} reserved bytes")
        if rank > MAX_RANK:
            raise MIMError("bad-shape", f"tensor {name!r} rank {rank}")
        dims = struct.unpack_from("<8Q", r, 16)
        if any(dims[rank:]):
            raise MIMError("reserved-non-zero", f"tensor {name!r} unused dims")
        off, ln = struct.unpack_from("<QQ", r, 80)
        dt = D.BY_ID.get(dt_id)
        if dt is None:
            raise MIMError("bad-dtype", f"tensor {name!r} dtype {dt_id}")
        shape = tuple(dims[:rank])
        size = D.byte_size(dt, shape)
        if size != ln:
            raise MIMError("size-mismatch", f"tensor {name!r}: shape and dtype need {size} bytes, entry says {ln}")
        return TensorInfo(name.decode("utf-8"), dt, shape, off, ln)

    def _check_tensors(self):
        prev = None
        spans = []
        self._names = []
        for i in range(self._ntensors):
            r = self._row(i)
            name = self._name(r)
            if not 0 < len(name) <= MAX_NAME_LENGTH:
                raise MIMError("bad-name", f"tensor {i} name")
            try:
                name.decode("utf-8")
            except UnicodeDecodeError:
                raise MIMError("bad-name", f"tensor {i} name") from None
            if prev is not None and prev >= name:
                if prev == name:
                    raise MIMError("duplicate", f"tensor {name!r}")
                raise MIMError("unsorted", f"tensor {name!r} after {prev!r}")
            prev = name
            ti = self._decode_row(r, name)
            if ti.offset % _tensor_alignment(ti.length):
                raise MIMError("misaligned", f"tensor {ti.name!r} offset {ti.offset}")
            if not _in_bounds(ti.offset, ti.length, self.preamble.data_length):
                raise MIMError("out-of-bounds", f"tensor {ti.name!r} data")
            spans.append((ti.offset, ti.length))
            self._names.append(name)
        spans.sort()
        end = 0
        for off, n in spans:
            want = _align_up(end, _tensor_alignment(n))
            if off != want:
                if off < end:
                    raise MIMError("overlap", f"tensor at {off} overlaps the previous one ending at {end}")
                raise MIMError("non-canonical-layout", f"tensor at {off}, expected {want}")
            end = off + n
        if end != self.preamble.data_length:
            raise MIMError("non-canonical-layout", f"weights section is {self.preamble.data_length} bytes, tensors end at {end}")

    # access --------------------------------------------------------------

    def section(self, kind):
        for s in self.sections:
            if s.kind == kind:
                return s
        return None

    def section_bytes(self, s):
        return self._data[s.offset : s.offset + s.length]

    def __len__(self):
        return self._ntensors

    def tensor_at(self, i):
        r = self._row(i)
        return self._decode_row(r, self._name(r))

    def tensors(self):
        return [self.tensor_at(i) for i in range(self._ntensors)]

    def tensor(self, name):
        """Looks a tensor up by name with a binary search, or returns None."""
        key = name.encode("utf-8")
        lo, hi = 0, self._ntensors
        while lo < hi:
            mid = (lo + hi) // 2
            if self._names[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        if lo < self._ntensors and self._names[lo] == key:
            return self.tensor_at(lo)
        return None

    def tensor_data(self, t):
        """The tensor's bytes, without copying."""
        base = self.preamble.data_offset + t.offset
        return self._data[base : base + t.length]

    def verify_hashes(self):
        for s in self.sections:
            if s.hash_alg == 1 and hashlib.sha256(self.section_bytes(s)).digest() != s.hash:
                raise MIMError("hash-mismatch", f"section kind {s.kind}")

    def metadata(self):
        from .metadata import decode_metadata

        s = self.section(METADATA)
        return decode_metadata(self.section_bytes(s)) if s else {}

    def graph(self):
        from .graph import decode_graph

        s = self.section(GRAPH)
        return decode_graph(self.section_bytes(s)) if s else None


def parse(data):
    return File(data)


def open_file(path):
    return File.open(path)

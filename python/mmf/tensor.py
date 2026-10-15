"""Dense tensors: a dtype, a shape and the elements in a NumPy array of the
dtype's storage type (see dtypes.py). Block-dtype tensors keep their raw
bytes."""

import numpy as np

from .dtypes import BY_NAME, byte_size


class Tensor:
    __slots__ = ("dtype", "shape", "arr", "raw")

    def __init__(self, dtype, shape, arr=None, raw=None):
        self.dtype = dtype
        self.shape = tuple(int(s) for s in shape)
        if dtype.is_block:
            if raw is None:
                raise ValueError("a block-dtype tensor needs raw bytes")
            if len(raw) != byte_size(dtype, self.shape):
                raise ValueError(f"{dtype.name}{list(self.shape)} needs {byte_size(dtype, self.shape)} bytes, got {len(raw)}")
            self.raw = raw
            self.arr = None
        else:
            a = np.asarray(arr, dtype=dtype.storage)
            if a.shape != self.shape:
                a = a.reshape(self.shape)
            self.arr = a
            self.raw = None

    @classmethod
    def from_bytes(cls, dtype, shape, data):
        """Reads packed little-endian element bytes, as stored in a file.
        Data is not copied when the layout allows."""
        if isinstance(dtype, str):
            dtype = BY_NAME[dtype]
        shape = tuple(int(s) for s in shape)
        if dtype.is_block:
            return cls(dtype, shape, raw=data)
        n = byte_size(dtype, shape)
        if len(data) != n:
            raise ValueError(f"{dtype.name}{list(shape)} needs {n} bytes, got {len(data)}")
        if dtype.kind == "bool":
            a = np.frombuffer(data, dtype=np.uint8).reshape(shape) != 0
        else:
            a = np.frombuffer(data, dtype=np.dtype(dtype.storage).newbyteorder("<")).reshape(shape)
        return cls(dtype, shape, a)

    def tobytes(self):
        if self.dtype.is_block:
            return bytes(self.raw)
        a = self.arr
        if self.dtype.kind == "bool":
            a = a.astype(np.uint8)
        return np.ascontiguousarray(a, dtype=np.dtype(a.dtype).newbyteorder("<")).tobytes()

    @property
    def size(self):
        return int(np.prod(self.shape, dtype=np.int64))

    def __repr__(self):
        return f"Tensor({self.dtype.name}{list(self.shape)})"


def bits_view(a):
    """A view of a storage array as unsigned integers of the same width, for
    operations that copy elements bit-exactly."""
    if a.dtype == np.bool_:
        return a
    return a.view(np.dtype(f"u{a.dtype.itemsize}"))

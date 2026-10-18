"""A reader and executor for mim files, written from the specification.

It is the second runtime of the project: it shares no code with the Go
reference implementation, and checks itself against the same conformance
and compatibility corpora. It also writes files (mim.writer) and exports
PyTorch models (mim.pytorch, which needs PyTorch)."""

from .errors import MIMError, RunError
from .tensor import Tensor
from . import dtypes

__all__ = ["MIMError", "RunError", "Tensor", "dtypes"]

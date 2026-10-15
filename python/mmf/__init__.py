"""A reader and executor for mmf files, written from the specification.

It is the second runtime of the project: it shares no code with the Go
reference implementation, and checks itself against the same conformance
and compatibility corpora."""

from .errors import MMFError, RunError
from .tensor import Tensor
from . import dtypes

__all__ = ["MMFError", "RunError", "Tensor", "dtypes"]

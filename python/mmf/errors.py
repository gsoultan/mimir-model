"""Error codes of spec/container.md section 9."""


class MMFError(Exception):
    """A rejected file or graph. `code` is one of the strings of
    spec/container.md section 9, so results can be compared with any other
    reader."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class RunError(Exception):
    """A run that cannot complete: a missing input, a shape mismatch, an
    unsupported composite (spec/primitives.md section 3.8)."""

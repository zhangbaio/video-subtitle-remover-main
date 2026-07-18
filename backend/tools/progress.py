"""Progress helpers that also work in ``pythonw`` processes."""

from __future__ import annotations

import sys
from typing import Any, TextIO

from tqdm import tqdm


class _NullTextStream:
    """Minimal writable stream which deliberately discards console output."""

    encoding = "utf-8"

    @staticmethod
    def write(value: str) -> int:
        return len(value)

    @staticmethod
    def flush() -> None:
        return None

    @staticmethod
    def isatty() -> bool:
        return False


_NULL_TEXT_STREAM = _NullTextStream()


def safe_console_stream(preferred: TextIO | None = None) -> TextIO:
    """Return a writable console stream, or a non-buffering null stream."""

    candidates = (
        preferred,
        getattr(sys, "__stdout__", None),
        getattr(sys, "stdout", None),
        getattr(sys, "__stderr__", None),
        getattr(sys, "stderr", None),
    )
    for stream in candidates:
        if stream is not None and callable(getattr(stream, "write", None)):
            return stream
    return _NULL_TEXT_STREAM  # type: ignore[return-value]


class SafeTqdm(tqdm):
    """``tqdm`` variant whose instance ``write`` is safe without a console."""

    def write(
        self,
        value: str,
        file: TextIO | None = None,
        end: str = "\n",
        nolock: bool = False,
    ) -> None:
        stream = file if file is not None else getattr(self, "fp", None)
        tqdm.write(
            value,
            file=safe_console_stream(stream),
            end=end,
            nolock=nolock,
        )


def safe_tqdm_write(
    value: str,
    file: TextIO | None = None,
    end: str = "\n",
    nolock: bool = False,
) -> None:
    """Write through ``tqdm`` without relying on a standard console stream."""

    tqdm.write(
        value,
        file=safe_console_stream(file),
        end=end,
        nolock=nolock,
    )


def safe_tqdm(*args: Any, file: TextIO | None = None, **kwargs: Any) -> SafeTqdm:
    """Create an enabled progress bar backed by a safe writable stream."""

    return SafeTqdm(*args, file=safe_console_stream(file), **kwargs)

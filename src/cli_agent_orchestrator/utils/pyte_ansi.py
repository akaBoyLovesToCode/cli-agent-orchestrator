"""Reconstruct ANSI-styled rows from a pyte screen buffer.

The pyte screen composites the raw byte stream into a cell grid and KEEPS
per-character styling: ``screen.buffer[y][x]`` is a
``Char(data, fg, bg, bold, italics, …)``. A 256-colour SGR (``38;5;N``)
becomes the colour's hex string (see :data:`pyte.graphics.FG_BG_256`, which
is invertible), truecolour stays a 6-digit hex string, and bold/italics are
flags. (SGR 2 dim carries no pyte cell state and is lost, but nothing in the
styled classifiers keys on it.) Re-emitting those styles as SGR runs recovers
the renderer evidence the styled Kimi Code classifier keys on — answer colour
253, tool-name bold 111, reasoning grey 244 + italic, user echo 222 — which
the escape-free ``screen.display`` collapses to plain text.
"""

import re
from typing import TYPE_CHECKING, List, Optional

from pyte.graphics import FG_BG_256

if TYPE_CHECKING:
    import pyte

#: pyte stores a 256-colour foreground as its hex string; invert the table.
_HEX_TO_INDEX = {hex_value: index for index, hex_value in enumerate(FG_BG_256)}

#: Reconstructed rows contain only SGR sequences, so a plain SGR strip turns
#: them back into escape-free text. Same shape as kimi_transcript's _SGR_RE;
#: kept local so the StatusMonitor does not import a provider module for a
#: generic escape operation.
_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_sgr(line: str) -> str:
    """Remove SGR sequences from a reconstructed row."""

    return _SGR_RE.sub("", line)


def _fg_sgr(fg: str) -> Optional[str]:
    """The foreground SGR run for a pyte cell colour, or None for default."""

    if fg == "default":
        return None
    if fg in _HEX_TO_INDEX:
        return f"\x1b[38;5;{_HEX_TO_INDEX[fg]}m"
    if len(fg) == 6:
        return f"\x1b[38;2;{int(fg[0:2], 16)};{int(fg[2:4], 16)};{int(fg[4:6], 16)}m"
    return None


def row_to_ansi(screen: "pyte.Screen", y: int) -> str:
    """Rebuild one screen row as ANSI text, preserving renderer styling.

    Style changes emit a reset-then-reapply run so downstream classifiers see
    the same SGR shapes the TUI originally drew. A row that starts in the
    default style starts with NO escape at all — a leading reset would break
    line-anchored patterns (``^\\s*``) that expect the glyph at the margin.
    """

    out: List[str] = []
    current = None
    for x in range(screen.columns):
        cell = screen.buffer[y][x]
        style = (_fg_sgr(cell.fg), cell.bold, cell.italics)
        if style != current:
            if style == (None, False, False):
                if current is not None:
                    out.append("\x1b[0m")
            else:
                if current is not None and current != (None, False, False):
                    out.append("\x1b[0m")
                if style[0]:
                    out.append(style[0])
                if style[1]:
                    out.append("\x1b[1m")
                if style[2]:
                    out.append("\x1b[3m")
            current = style
        out.append(cell.data)
    out.append("\x1b[0m")
    return "".join(out).rstrip()


def screen_to_ansi_rows(screen: "pyte.Screen") -> List[str]:
    """Rebuild every viewport row as ANSI text (see :func:`row_to_ansi`)."""

    return [row_to_ansi(screen, y) for y in range(screen.lines)]

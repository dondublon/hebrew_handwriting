"""Minimal bidirectional text handling.

The Pillow build that is available in this project has no ``libraqm``, so
``ImageDraw.text(..., direction="rtl")`` is not supported and Hebrew strings
would be drawn in *logical* order (i.e. mirrored).  To render a realistic
Hebrew line we therefore reorder the string into *visual* order ourselves.

For the strings this project works with (Hebrew letters, digits, ASCII
punctuation and spaces) the reordering function is its own inverse, which is
verified by ``scripts/selftest.py``.  A full UAX#9 implementation is not
needed - and niqqud, which would need one, is out of scope for the PoC.
"""

from __future__ import annotations

from .charset import HEBREW_SET

# U+05F3 geresh and U+05F4 gershayim behave like Hebrew letters.
RTL_ONLY = frozenset(HEBREW_SET | {"\u05f3", "\u05f4"})

MIRROR = {
    "(": ")",
    ")": "(",
    "[": "]",
    "]": "[",
    "{": "}",
    "}": "{",
    "<": ">",
    ">": "<",
}

RTL = "R"
LTR = "L"
NEUTRAL = "N"


def char_class(ch: str) -> str:
    if ch in RTL_ONLY:
        return RTL
    if ch.isdigit() or ("a" <= ch.lower() <= "z"):
        return LTR
    return NEUTRAL


def split_runs(text: str) -> list[tuple[str, str]]:
    """Split ``text`` into maximal runs of equal direction, then resolve the
    neutral runs (Unicode bidi rules N1/N2 with an RTL paragraph direction)
    and merge the result.

    Resolving neutrals against the paragraph direction - instead of gluing
    them to the previous run - is what makes the transformation symmetrical
    for strings that mix Hebrew and digits (e.g. ``"מתכנת 711"``).
    """
    runs: list[list] = []
    previous_is_ltr = False
    for ch in text:
        cls = char_class(ch)
        # A mark directly after a digit/Latin run stays glued to it ("1.", "12:"):
        # that is how numbered items look in the documents, and gluing it keeps
        # the transformation involutive.  Resolving such a mark against the
        # paragraph direction instead (UAX#9 N1) makes the forward and the
        # backward pass disagree, which `selftest` catches.
        if cls == NEUTRAL and ch.strip() and previous_is_ltr:
            cls = LTR
        if runs and runs[-1][0] == cls:
            runs[-1][1].append(ch)
        else:
            runs.append([cls, [ch]])
        previous_is_ltr = cls == LTR

    for i, run in enumerate(runs):
        if run[0] != NEUTRAL:
            continue
        prev = runs[i - 1][0] if i > 0 else None
        nxt = runs[i + 1][0] if i + 1 < len(runs) else None
        if prev is not None and prev == nxt:
            run[0] = prev
        elif prev is not None and nxt is not None:
            run[0] = RTL  # N2: paragraph direction (Hebrew)
        else:
            run[0] = prev or nxt or RTL

    merged: list[tuple[str, str]] = []
    for cls, chars in runs:
        if merged and merged[-1][0] == cls:
            merged[-1] = (cls, merged[-1][1] + "".join(chars))
        else:
            merged.append((cls, "".join(chars)))
    return merged



def logical_to_visual(text: str) -> str:
    """Reorder logical (typing) order into visual (left-to-right) order.

    ``logical_to_visual`` is an involution for the character classes used by
    this project, so it doubles as ``visual_to_logical``.
    """
    runs = split_runs(text)
    out: list[str] = []
    for cls, run in reversed(runs):
        if cls == RTL:
            out.append("".join(MIRROR.get(ch, ch) for ch in reversed(run)))
        else:
            out.append("".join(MIRROR.get(ch, ch) for ch in run))
    return "".join(out)


def visual_to_logical(text: str) -> str:
    """Inverse of :func:`logical_to_visual` (same operation, see docstring)."""
    return logical_to_visual(text)

"""Alphabet definition and label (de)serialisation.

Index 0 is reserved for the CTC blank.  Everything else follows the order of
``CHARS``; the resulting mapping is stored inside the checkpoint, therefore it
must never be reordered without retraining.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Alef, bet, ... tav are the 22 "regular" letters, the 5 final forms (kaf,
# mem, nun, pe, tsadi) are listed separately because a number of checks below
# depend on that distinction.
REGULAR_LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"
FINAL_LETTERS = "ךםןףץ"
HEBREW_LETTERS = REGULAR_LETTERS + FINAL_LETTERS
DIGITS = "0123456789"
PUNCTUATION = ".,:;?!'\"()-"
SPACE = " "

CHARS = SPACE + HEBREW_LETTERS + DIGITS + PUNCTUATION

BLANK_INDEX = 0
BLANK = "<blank>"

CHAR_TO_IDX: dict[str, int] = {ch: i + 1 for i, ch in enumerate(CHARS)}
IDX_TO_CHAR: dict[int, str] = {i + 1: ch for i, ch in enumerate(CHARS)}

HEBREW_SET = frozenset(HEBREW_LETTERS)
FINAL_SET = frozenset(FINAL_LETTERS)
# Characters that are normally written inside a word.
WORD_CHARS = frozenset(REGULAR_LETTERS + DIGITS)


def num_classes() -> int:
    """Number of outputs of the network (including the CTC blank)."""
    return len(CHARS) + 1


def encode(visual_text: str, drop_unknown: bool = True) -> list[int]:
    """Map a *visual order* string to a list of class indices."""
    if drop_unknown:
        return [CHAR_TO_IDX[ch] for ch in visual_text if ch in CHAR_TO_IDX]
    return [CHAR_TO_IDX[ch] for ch in visual_text]


def decode_indices(indices: Iterable[int]) -> str:
    """Map class indices back to a *visual order* string."""
    return "".join(IDX_TO_CHAR.get(int(i), "") for i in indices)


def is_supported(text: str) -> bool:
    return all(ch in CHAR_TO_IDX for ch in text)


def validate_lexicon(words: Iterable[str]) -> list[str]:
    """Return a list of human readable problems found in a word list."""
    problems: list[str] = []
    for word in words:
        for ch in word:
            if ch not in CHAR_TO_IDX:
                problems.append(f"{word!r}: unsupported character {ch!r}")
        for i, ch in enumerate(word):
            if ch in FINAL_SET and i != len(word) - 1:
                problems.append(f"{word!r}: final letter {ch!r} not at the end")
    return problems


def coverage_report(charset_available: Sequence[str] | set[str]) -> dict[str, list[str]]:
    """Compare the model alphabet with the characters a font can render."""
    available = set(charset_available)
    return {
        "missing_in_font": [c for c in CHARS if c not in available],
        "extra_in_font": sorted(available - set(CHARS)),
    }

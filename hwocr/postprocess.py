"""Optional lexicon-based post-processing.

OCR engines normally use a dictionary: Tesseract ships one, and the same idea
helps here - a recognised word that is not in the word list but is one or two
edits away from a word that *is* is usually a small mistake (a dropped
diacritic-like stroke, an א/ה or ב/כ confusion, a missing final form).

This is deliberately a *post-processing* step, not part of the recogniser:

* it is off by default (``--lexicon-decode`` turns it on),
* it can only help when the text really comes from a known vocabulary; for
  arbitrary words it may "correct" them into dictionary words, which is why
  the effect is reported separately in the evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .metrics import edit_distance


class WordCorrector:
    """Replaces unknown words by their closest lexicon entries."""

    PREFIXES = "הובלמכש"  # one-letter prefixes of Hebrew words

    def __init__(self, words: Iterable[str], max_relative_distance: float = 0.34,
                 max_absolute_distance: int = 3, min_length: int = 3) -> None:
        self.words = sorted({w for w in words if w})
        self.lexicon = set(self.words)
        self.max_relative_distance = float(max_relative_distance)
        self.max_absolute_distance = int(max_absolute_distance)
        self.min_length = int(min_length)
        self._by_length: dict[int, list[str]] = defaultdict(list)
        for word in self.words:
            self._by_length[len(word)].append(word)

    def threshold(self, word: str) -> int:
        return min(self.max_absolute_distance,
                   max(1, int(len(word) * self.max_relative_distance)))

    def correct_word(self, word: str) -> str:
        """Returns the corrected word (or the input when nothing fits)."""
        if len(word) < self.min_length or word in self.lexicon:
            return word
        # Hebrew words take one-letter prefixes (ה, ו, ב, ל, מ, כ, ש); do not
        # "correct" away a prefix of a word that is in the list ('החתול').
        if word[0] in self.PREFIXES and len(word) > 3 and word[1:] in self.lexicon:
            return word
        limit = self.threshold(word)
        best = word
        best_key: tuple[int, int] | None = None
        for length in range(max(1, len(word) - limit), len(word) + limit + 1):
            for candidate in self._by_length.get(length, ()):
                distance = edit_distance(word, candidate)
                if distance > limit:
                    continue
                # prefer the closest match, then the one that keeps the length
                # (most mistakes are single characters, not dropped letters)
                key = (distance, abs(length - len(word)))
                if best_key is None or key < best_key:
                    best, best_key = candidate, key
                    if key == (1, 0):
                        return best
        return best

    def correct_text(self, text: str) -> str:
        """Correct word by word, keeping the original separators."""
        out: list[str] = []
        word: list[str] = []

        def flush() -> None:
            if word:
                out.append(self.correct_word("".join(word)))
                word.clear()

        for ch in text:
            if ch == " ":
                flush()
                out.append(ch)
            else:
                word.append(ch)
        flush()
        return "".join(out)


def correct_words_in_report(pairs: Sequence[tuple[str, str]], corrector: WordCorrector,
                            lexical_words: Iterable[str] | None = None) -> list[tuple[str, str]]:
    """Helper used by the evaluation to score the corrected output."""
    lexical = set(lexical_words) if lexical_words else corrector.lexicon
    result = []
    for reference, hypothesis in pairs:
        if reference.split() and all(word in lexical for word in reference.split()):
            result.append((reference, corrector.correct_text(hypothesis)))
        else:
            result.append((reference, hypothesis))
    return result

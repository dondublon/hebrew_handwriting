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

from . import charset
from .metrics import edit_distance


class WordCorrector:
    """Replaces unknown words by their closest lexicon entries."""

    PREFIXES = "הובלמכש"  # one-letter prefixes of Hebrew words
    # Marks that may surround a word: they are peeled off before the word is
    # corrected and put back afterwards, so post-processing never eats the
    # punctuation the recogniser produced ('הם:' stays 'הם:').  The hyphen is
    # *not* here - it also occurs inside a word (see ``_correct_token``).
    MARKS = charset.PUNCTUATION.replace("-", "")

    def __init__(self, words: Iterable[str], max_relative_distance: float = 0.34,
                 max_absolute_distance: int = 3, min_length: int = 3,
                 segment: bool = True, min_part: int = 2) -> None:
        self.words = sorted({w for w in words if w})
        self.lexicon = set(self.words)
        self.max_relative_distance = float(max_relative_distance)
        self.max_absolute_distance = int(max_absolute_distance)
        self.min_length = int(min_length)
        self.segment = bool(segment)
        self.min_part = int(min_part)
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
        return self._matches(word) or word

    def correct_text(self, text: str) -> str:
        """Full post-processing: word segmentation, then per-word correction."""
        if self.segment:
            text = self.segment_text(text)
        out: list[str] = []
        word: list[str] = []

        def flush() -> None:
            if word:
                out.append(self._correct_token("".join(word)))
                word.clear()

        for ch in text:
            if ch == " ":
                flush()
                out.append(ch)
            else:
                word.append(ch)
        flush()
        return "".join(out)

    # ------------------------------------------------------------------
    def _split_marks(self, token: str) -> tuple[str, str, str]:
        """Peels the surrounding marks off a token: (prefix, core, suffix)."""
        start = 0
        while start < len(token) and token[start] in self.MARKS:
            start += 1
        end = len(token)
        while end > start and token[end - 1] in self.MARKS:
            end -= 1
        return token[:start], token[start:end], token[end:]

    def _correct_token(self, token: str) -> str:
        """Corrects one whitespace-separated token, keeping its marks.

        ``הם:`` -> the core ``הם`` is looked up, the colon stays.  A hyphenated
        compound is corrected part by part (``קול-מוזיקה``), a lone ``-`` is
        left untouched, so the marks the recogniser produced survive.
        """
        prefix, core, suffix = self._split_marks(token)
        if not core:
            return token
        parts = [self.correct_word(part) for part in core.split("-") if part]
        if parts:
            core = "-".join(parts)
        return prefix + core + suffix

    def _matches(self, word: str) -> str | None:
        """Closest lexicon word within the correction threshold (``None`` if none)."""
        if not word:
            return None
        if word in self.lexicon:
            return word
        if len(word) < self.min_length:
            return None
        limit = self.threshold(word)
        best: str | None = None
        best_key: tuple[int, int] | None = None
        for length in range(max(1, len(word) - limit), len(word) + limit + 1):
            for candidate in self._by_length.get(length, ()):
                distance = edit_distance(word, candidate)
                if distance > limit:
                    continue
                # не «съедаем» буквы: замены допустимы, а вот пропуск/вставка
                # больше одной буквы — уже другая информация, такое отвергаем
                if distance > 1 and length != len(word):
                    continue
                key = (distance, abs(length - len(word)))
                if best_key is None or key < best_key:
                    best, best_key = candidate, key
                    if key == (1, 0):
                        return best
        return best

    def _parts(self, word: str) -> list[list[str]]:
        """All ways to split ``word`` into lexicon words (prefix + 1..3 parts).

        Parts may be one edit away from a lexicon word (``התחשות`` ~
        ``התרחשות``): the per-word correction fixes them afterwards, which is
        what makes splitting work on noisy recognition output.
        """
        options: list[list[str]] = []
        # приставка + слово: האנגיה -> ה + אנגיה
        if len(word) > 2 and word[0] in self.PREFIXES and self._matches(word[1:]):
            options.append([word[0], word[1:]])
        length = len(word)
        for i in range(self.min_part, length - self.min_part + 1):
            first, rest = word[:i], word[i:]
            if self._matches(first) and self._matches(rest):
                options.append([first, rest])
            elif not self._matches(rest):
                for j in range(1, len(rest) - self.min_part + 1):
                    middle, last = rest[:j], rest[j:]
                    if self._matches(first) and self._matches(middle) and self._matches(last):
                        options.append([first, middle, last])
        # предпочитаем: все части точно в словаре -> меньше частей ->
        # самый длинный последний фрагмент (в иврите приставки ה/ו/ב/ל/מ/כ/ש
        # принадлежат следующему слову: «סוגי | האנגיה» лучше, чем «סוגיה | אנגיה»)
        # -> более ровное деление
        options.sort(key=lambda parts: (
            sum(1 for part in parts if part not in self.lexicon),
            len(parts), -len(parts[-1]), max(len(p) for p in parts)))
        return options

    def segment_text(self, text: str) -> str:
        """Split words that are glued together and merge words split by mistake.

        Only tokens that are *not* in the word list are touched, so already
        correct text is left alone.  This is what fixes a missing space when two
        written words touch each other (no ink gap for the recogniser to see).
        """
        words = text.split(" ")
        changed = True
        rounds = 0
        while changed and rounds < 3:
            changed = False
            rounds += 1
            merged: list[str] = []
            i = 0
            while i < len(words):
                if i + 1 < len(words) and words[i] and words[i + 1]:
                    candidate = words[i] + words[i + 1]
                    if (words[i] not in self.lexicon and words[i + 1] not in self.lexicon
                            and candidate in self.lexicon):
                        merged.append(candidate)
                        i += 2
                        changed = True
                        continue
                merged.append(words[i])
                i += 1
            words = merged

            split_words: list[str] = []
            for word in words:
                prefix, core, suffix = self._split_marks(word)
                if len(core) >= 2 * self.min_part and not self._matches(core):
                    options = self._parts(core)
                    if options:
                        # сами части корректируются дальше, по словам
                        best = options[0]
                        if len(best) > 1:
                            parts = [self._matches(part) or part for part in best]
                            # знаки остаются на своих местах
                            parts[0] = prefix + parts[0]
                            parts[-1] = parts[-1] + suffix
                            split_words.extend(parts)
                            changed = True
                            continue
                split_words.append(word)
            words = split_words
        return " ".join(words)



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

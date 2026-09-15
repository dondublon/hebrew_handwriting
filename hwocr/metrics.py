"""Error metrics (character and word error rate)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Levenshtein distance with unit costs (iterative, O(min(n, m)) memory)."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_ch in enumerate(reference, start=1):
        current = [i]
        for j, hyp_ch in enumerate(hypothesis, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (ref_ch != hyp_ch)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> tuple[int, int]:
    """Returns (errors, reference length) for a single pair."""
    return edit_distance(reference, hypothesis), max(1, len(reference))


def wer(reference: str, hypothesis: str) -> tuple[int, int]:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return edit_distance(ref_words, hyp_words), max(1, len(ref_words))


@dataclass
class Report:
    """Aggregated quality report over a dataset."""

    samples: int = 0
    char_errors: int = 0
    char_total: int = 0
    word_errors: int = 0
    word_total: int = 0
    exact_matches: int = 0
    confidence_sum: float = 0.0

    def add(self, reference: str, hypothesis: str, confidence: float = 0.0) -> None:
        err, total = cer(reference, hypothesis)
        werr, wtotal = wer(reference, hypothesis)
        self.samples += 1
        self.char_errors += err
        self.char_total += total
        self.word_errors += werr
        self.word_total += wtotal
        self.exact_matches += int(reference == hypothesis)
        self.confidence_sum += confidence

    @property
    def cer(self) -> float:
        return self.char_errors / self.char_total if self.char_total else 0.0

    @property
    def wer(self) -> float:
        return self.word_errors / self.word_total if self.word_total else 0.0

    @property
    def accuracy(self) -> float:
        return 1.0 - self.cer

    @property
    def line_accuracy(self) -> float:
        return self.exact_matches / self.samples if self.samples else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.samples if self.samples else 0.0

    def merge(self, other: "Report") -> "Report":
        for name in ("samples", "char_errors", "char_total", "word_errors",
                     "word_total", "exact_matches"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.confidence_sum += other.confidence_sum
        return self

    def summary(self) -> str:
        return (f"n={self.samples} CER={self.cer:.4f} WER={self.wer:.4f} "
                f"line_acc={self.line_accuracy:.3f} conf={self.mean_confidence:.3f}")


def report_from_pairs(pairs: Iterable[tuple[str, str]]) -> Report:
    report = Report()
    for reference, hypothesis in pairs:
        report.add(reference, hypothesis)
    return report

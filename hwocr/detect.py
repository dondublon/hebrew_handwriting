"""Layout analysis: find and crop text lines on a page.

Deliberately dependency-free (OpenCV only): binarise, deskew by maximising the
sharpness of the horizontal ink profile, group connected components into lines
by vertical overlap, then sort the lines top-to-bottom.  This is enough for
rendered/photographed pages of handwriting-style text and keeps the proof of
concept self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import augment


@dataclass
class LineBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, image: np.ndarray, pad: int = 2) -> np.ndarray:
        y0 = max(0, self.y0 - pad)
        x0 = max(0, self.x0 - pad)
        y1 = min(image.shape[0], self.y1 + pad)
        x1 = min(image.shape[1], self.x1 + pad)
        return image[y0:y1, x0:x1]

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink -> 255, background -> 0 (works for both polarities)."""
    gray = augment.ensure_gray(gray)
    if float(gray.mean()) < 127:  # light text on dark background
        gray = 255 - gray
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.bitwise_or(binary, otsu)


def estimate_skew(binary: np.ndarray, max_angle: float = 8.0) -> float:
    """Projection-profile skew search: the sharpest profile wins."""
    best_angle, best_score = 0.0, -1.0
    h, w = binary.shape
    center = (w / 2, h / 2)
    for angle in np.arange(-max_angle, max_angle + 0.5, 2.0):
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST)
        profile = rotated.sum(axis=1, dtype=np.float64)
        score = float(np.square(np.diff(profile)).sum())
        if score > best_score:
            best_score, best_angle = score, float(angle)
    for angle in np.arange(best_angle - 1.5, best_angle + 1.5, 0.5):
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST)
        profile = rotated.sum(axis=1, dtype=np.float64)
        score = float(np.square(np.diff(profile)).sum())
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.25:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = cv2.BORDER_REPLICATE if image.ndim == 3 else cv2.BORDER_CONSTANT
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=border, borderValue=255)


def components(binary: np.ndarray, min_area: int = 8) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the ink blobs (noise filtered)."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area or w <= 0 or h <= 0:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def _cluster_lines(boxes: list[tuple[int, int, int, int]],
                   min_overlap: float = 0.45) -> list[list[tuple[int, int, int, int]]]:
    """Greedy vertical-overlap clustering (handles Hebrew letters and their dots)."""
    clusters: list[dict] = []
    for box in sorted(boxes, key=lambda b: b[1]):
        y0, y1 = box[1], box[3]
        best = None
        for cluster in clusters:
            top, bottom = cluster["y0"], cluster["y1"]
            overlap = min(y1, bottom) - max(y0, top)
            if overlap > min_overlap * min(y1 - y0, bottom - top):
                best = cluster
                break
        if best is None:
            clusters.append({"y0": y0, "y1": y1, "boxes": [box]})
        else:
            best["boxes"].append(box)
            best["y0"] = min(best["y0"], y0)
            best["y1"] = max(best["y1"], y1)
    return [cluster["boxes"] for cluster in clusters]


def segment_lines(
    image: np.ndarray,
    min_height: int = 8,
    min_ink: int = 8,
    pad: int = 3,
    max_side: int = 1600,
) -> list[LineBox]:
    """Detect text lines on a page; returns boxes sorted top-to-bottom."""
    gray = augment.ensure_gray(image)
    binary = binarize(gray)

    # Skew correction on a downscaled copy (cheap and accurate enough).
    scale = 1.0
    if max(binary.shape) > max_side:
        scale = max_side / max(binary.shape)
        small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = binary
    angle = estimate_skew(small)
    if abs(angle) >= 0.25:
        gray = deskew(gray, angle)
        binary = binarize(gray)

    boxes = components(binary, min_area=min_ink)
    lines: list[LineBox] = []
    for cluster in _cluster_lines(boxes):
        x0 = min(b[0] for b in cluster)
        y0 = min(b[1] for b in cluster)
        x1 = max(b[2] for b in cluster)
        y1 = max(b[3] for b in cluster)
        if (y1 - y0) < min_height:
            continue
        lines.append(LineBox(max(0, x0 - pad), max(0, y0 - pad),
                             min(gray.shape[1], x1 + pad), min(gray.shape[0], y1 + pad)))
    lines.sort(key=lambda box: (box.y0, box.x0))
    return lines


def segment_words(line_image: np.ndarray, gap_factor: float = 0.45) -> list[tuple[int, int]]:
    """Column ranges of the words inside a single line image (debug helper)."""
    binary = binarize(line_image)
    columns = (binary > 0).sum(axis=0)
    height = max(1, line_image.shape[0])
    threshold = max(3, int(gap_factor * height))
    gaps: list[tuple[int, int]] = []
    start = None
    for x, value in enumerate(columns):
        if value == 0 and start is None:
            start = x
        elif value > 0 and start is not None:
            if x - start >= threshold:
                gaps.append((start, x))
            start = None
    spans: list[tuple[int, int]] = []
    cursor = 0
    for gap_start, gap_end in gaps:
        if gap_start > cursor:
            spans.append((cursor, gap_start))
        cursor = gap_end
    if cursor < len(columns):
        spans.append((cursor, len(columns)))
    return [(a, b) for a, b in spans if b - a > 1]


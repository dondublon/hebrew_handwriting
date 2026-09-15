"""Image level augmentations.

Two pipelines are provided:

* ``apply_light``  - small geometric/photometric jitter, keeps the sample close
  to a clean computer rendering (the main training distribution).
* ``apply_heavy``  - a "photo of a page taken with a phone" simulation
  (perspective, shadow, motion blur, noise, JPEG artefacts).  Samples of this
  kind are mixed into training so that the model survives real images.

``to_model_input`` performs the final normalisation; because it crops to the
ink and scales by height, it is also exactly what inference uses.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

WHITE = 255
"""Background value used everywhere in the pipeline (ink is dark)."""


def _np_rng(rng: random.Random) -> np.random.Generator:
    """Deterministic numpy generator derived from a ``random.Random``.

    Keeps every sample reproducible from its seed while allowing fast array
    randomness (noise, elastic fields).
    """
    return np.random.default_rng(rng.getrandbits(32))


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert any input image to uint8 grayscale with dark ink on light paper."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8, copy=False)


def _warp_border() -> int:
    return cv2.BORDER_CONSTANT


# ---------------------------------------------------------------------------
# geometric
# ---------------------------------------------------------------------------
def add_border(img: np.ndarray, rng: random.Random, px: tuple[int, int] = (2, 14)) -> np.ndarray:
    top, bottom = rng.randint(*px), rng.randint(*px)
    left, right = rng.randint(*px), rng.randint(*px)
    return cv2.copyMakeBorder(img, top, bottom, left, right, _warp_border(), value=WHITE)


def random_rotate(img: np.ndarray, rng: random.Random, max_angle: float = 2.5) -> np.ndarray:
    angle = rng.uniform(-max_angle, max_angle)
    if abs(angle) < 0.05:
        return img
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=_warp_border(), borderValue=WHITE)


def random_affine(img: np.ndarray, rng: random.Random, max_shear: float = 0.18,
                  max_scale: float = 0.08, max_shift: float = 0.04) -> np.ndarray:
    h, w = img.shape
    shear = rng.uniform(-max_shear, max_shear)
    scale = 1.0 + rng.uniform(-max_scale, max_scale)
    dx, dy = rng.uniform(-max_shift, max_shift) * w, rng.uniform(-max_shift, max_shift) * h
    matrix = np.array([[scale, shear * scale, dx],
                       [rng.uniform(-max_shear, max_shear) * scale * 0.5, scale, dy]],
                      dtype=np.float32)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=_warp_border(), borderValue=WHITE)


def random_perspective(img: np.ndarray, rng: random.Random, mag: float = 0.035) -> np.ndarray:
    h, w = img.shape
    mx, my = mag * w, mag * h * 1.5
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = np.float32([
        [rng.uniform(0, mx), rng.uniform(0, my)],
        [w - 1 - rng.uniform(0, mx), rng.uniform(0, my)],
        [rng.uniform(0, mx), h - 1 - rng.uniform(0, my)],
        [w - 1 - rng.uniform(0, mx), h - 1 - rng.uniform(0, my)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=_warp_border(), borderValue=WHITE)


def elastic(img: np.ndarray, rng: random.Random, alpha: float = 2.5, sigma: float = 6.0) -> np.ndarray:
    """Smooth random displacement - mimics the unevenness of a hand written line."""
    h, w = img.shape
    gen = _np_rng(rng)
    dx = cv2.GaussianBlur((gen.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((gen.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, grid_x + dx, grid_y + dy, cv2.INTER_LINEAR,
                     borderMode=_warp_border(), borderValue=WHITE)


def downscale_upscale(img: np.ndarray, rng: random.Random, factor: tuple[float, float] = (1.0, 2.2)) -> np.ndarray:
    """Low resolution sensor / small scan."""
    f = rng.uniform(*factor)
    if f <= 1.05:
        return img
    h, w = img.shape
    small = cv2.resize(img, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# photometric
# ---------------------------------------------------------------------------
def random_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    kind = rng.random()
    if kind < 0.55:
        sigma = rng.uniform(0.3, 1.4)
        return cv2.GaussianBlur(img, (0, 0), sigma)
    if kind < 0.8:
        size = rng.choice([3, 5, 7])
        return cv2.medianBlur(img, size)
    # motion blur
    size = rng.choice([3, 5, 7, 9])
    kernel = np.zeros((size, size), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[size // 2, :] = 1.0
    else:
        np.fill_diagonal(kernel, 1.0)
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def random_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)
    gen = _np_rng(rng)
    shape = img.shape
    if rng.random() < 0.7:  # gaussian sensor noise
        out += gen.normal(0.0, rng.uniform(1.5, 12.0), shape).astype(np.float32)
    if rng.random() < 0.35:  # salt & pepper
        amount = rng.uniform(0.0005, 0.01)
        mask = gen.random(shape)
        out[mask < amount / 2] = 0.0
        out[mask > 1 - amount / 2] = 255.0
    if rng.random() < 0.25:  # speckle / dust
        amount = rng.uniform(0.0002, 0.002)
        out[gen.random(shape) < amount] = rng.uniform(0, 90)
    return np.clip(out, 0, 255).astype(np.uint8)


def uneven_illumination(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Gradients, vignetting and shadow cast onto the page."""
    h, w = img.shape
    out = img.astype(np.float32)
    low, high = rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.05)
    if rng.random() < 0.5:  # horizontal ramp
        gain = np.linspace(low, high, w, dtype=np.float32)[None, :]
    else:                   # vertical ramp
        gain = np.linspace(low, high, h, dtype=np.float32)[:, None]
    out *= gain
    if rng.random() < 0.5:  # vignette
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2 + 1e-6)) ** 2 + ((yy - h / 2) / (h / 2 + 1e-6)) ** 2)
        out *= 1.0 - rng.uniform(0.05, 0.3) * np.clip(r, 0, 1.5)
    if rng.random() < 0.45:  # a soft shadow band
        band = np.zeros((h, w), dtype=np.float32)
        pts = np.array([[rng.randint(-w, w), 0], [rng.randint(0, 2 * w), 0],
                        [rng.randint(0, 2 * w), h], [rng.randint(-w, w), h]], dtype=np.int32)
        cv2.fillPoly(band, [pts], 1.0)
        band = cv2.GaussianBlur(band, (0, 0), rng.uniform(6, 25))
        out *= 1.0 - rng.uniform(0.1, 0.4) * band
    return np.clip(out, 0, 255).astype(np.uint8)


def random_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = 1.0 + rng.uniform(-0.25, 0.35)   # contrast
    beta = rng.uniform(-25, 25)              # brightness
    gamma = 1.0 + rng.uniform(-0.35, 0.35)   # gamma
    out = img.astype(np.float32) / 255.0
    out = np.clip(out, 0, 1) ** gamma
    out = out * alpha * 255.0 + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def paper_texture(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Faint paper grain, dust and a couple of fold lines."""
    h, w = img.shape
    out = img.astype(np.float32)
    grain = cv2.GaussianBlur((_np_rng(rng).random((h, w)) * 255).astype(np.float32),
                             (0, 0), rng.uniform(1.5, 5.0))
    out = out * (1.0 - rng.uniform(0.03, 0.12)) + grain * rng.uniform(0.03, 0.12)
    for _ in range(rng.randint(0, 2)):  # pencil-thin scratches / folds
        x1, x2 = rng.randint(0, w), rng.randint(0, w)
        cv2.line(out, (x1, 0), (x2, h), float(rng.uniform(120, 235)), 1, cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)


def ink_thickness(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Pen pressure: thicker or thinner strokes."""
    size = rng.choice([1, 1, 2, 3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1))
    binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    binary = cv2.dilate(binary, kernel) if rng.random() < 0.5 else cv2.erode(binary, kernel)
    return np.where(binary > 0, img, WHITE).astype(np.uint8)


def jpeg_artifacts(img: np.ndarray, rng: random.Random, quality: tuple[int, int] = (35, 92)) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(*quality)])
    if not ok:  # pragma: no cover - cannot happen with valid input
        return img
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def invert(img: np.ndarray) -> np.ndarray:
    return (255 - img).astype(np.uint8)


# ---------------------------------------------------------------------------
# pipelines
# ---------------------------------------------------------------------------
def apply_light(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mild jitter: same domain as a clean font rendering."""
    if rng.random() < 0.80:
        img = random_rotate(img, rng, 2.0)
    if rng.random() < 0.70:
        img = random_affine(img, rng, 0.12, 0.05, 0.02)
    if rng.random() < 0.35:
        img = elastic(img, rng, 1.6, 7.0)
    if rng.random() < 0.50:
        img = random_contrast(img, rng)
    if rng.random() < 0.40:
        img = random_blur(img, rng)
    if rng.random() < 0.35:
        img = random_noise(img, rng)
    if rng.random() < 0.30:
        img = ink_thickness(img, rng)
    return add_border(img, rng, (1, 6))


def apply_heavy(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulates a photo of a page: perspective, light, grain, JPEG."""
    img = add_border(img, rng, (6, 24))
    if rng.random() < 0.80:
        img = random_perspective(img, rng, 0.030)
    if rng.random() < 0.70:
        img = random_rotate(img, rng, 2.5)
    if rng.random() < 0.60:
        img = random_affine(img, rng, 0.20, 0.10, 0.04)
    if rng.random() < 0.50:
        img = elastic(img, rng, 2.5, 6.0)
    if rng.random() < 0.40:
        img = downscale_upscale(img, rng)
    img = uneven_illumination(img, rng)
    img = paper_texture(img, rng)
    if rng.random() < 0.60:
        img = random_blur(img, rng)
    img = random_noise(img, rng)
    if rng.random() < 0.60:
        img = random_contrast(img, rng)
    if rng.random() < 0.35:
        img = ink_thickness(img, rng)
    if rng.random() < 0.12:  # dark mode screenshots etc.
        img = invert(img)
    if rng.random() < 0.60:
        img = jpeg_artifacts(img, rng)
    return img


def apply_augment(img: np.ndarray, rng: random.Random, heavy: bool = False) -> np.ndarray:
    return apply_heavy(img, rng) if heavy else apply_light(img, rng)


# ---------------------------------------------------------------------------
# model input
# ---------------------------------------------------------------------------
def crop_ink(img: np.ndarray, margin: int = 2, threshold: int = 200) -> np.ndarray:
    """Crop to the ink bounding box (keeps a small white margin).

    Falls back to a plain border-crop when the image is inverted
    (light ink on dark background).
    """
    binary = (img < threshold).astype(np.uint8)
    if binary.sum() < 4:
        binary = (img > threshold).astype(np.uint8)
    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    y0, y1 = max(0, rows[0] - margin), min(img.shape[0], rows[-1] + 1 + margin)
    x0, x1 = max(0, cols[0] - margin), min(img.shape[1], cols[-1] + 1 + margin)
    return img[y0:y1, x0:x1]


def prepare_for_model(
    img: np.ndarray,
    height: int = 40,
    max_width: int = 512,
    min_width: int = 16,
) -> tuple[np.ndarray, int]:
    """Crop, scale to ``height`` and pad; returns (uint8 image, valid width px)."""
    cropped = crop_ink(ensure_gray(img))
    h, w = cropped.shape[:2]
    scale = height / float(h)
    if w * scale > max_width:
        scale = max_width / float(w)
    new_w = max(min_width, int(round(w * scale)))
    new_h = max(1, min(height, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=interp)

    pad_w = int(np.ceil(new_w / 8.0) * 8)
    canvas = np.full((height, pad_w), WHITE, dtype=np.uint8)
    y = max(0, (height - new_h) // 2)
    canvas[y:y + new_h, :new_w] = resized
    return canvas, new_w


def normalize(img: np.ndarray) -> np.ndarray:
    """uint8 grayscale -> float32 in [-1, 1] (ink dark, background light)."""
    return (img.astype(np.float32) / 255.0 - 0.5) / 0.5


def to_model_input(
    img: np.ndarray,
    height: int = 40,
    max_width: int = 512,
    min_width: int = 16,
) -> tuple[np.ndarray, int]:
    """Full pre-processing: returns (float32 array of shape (1, H, W), valid width)."""
    canvas, valid_w = prepare_for_model(img, height, max_width, min_width)
    return normalize(canvas)[None, ...], valid_w



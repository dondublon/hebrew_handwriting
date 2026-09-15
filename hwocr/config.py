"""Central configuration for the Hebrew handwriting OCR proof-of-concept."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = PROJECT_ROOT / "fonts"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_CHECKPOINT = MODEL_DIR / "crnn_gveret.pt"


@dataclass
class Config:
    """All tunables in one place.

    The dataclass is serialised next to every checkpoint, so that inference
    always uses exactly the architecture/pre-processing of the training run.
    """

    # ---- synthetic data -------------------------------------------------
    img_height: int = 40          # height of the model input (px)
    max_width: int = 512          # hard cap for the model input width (px)
    font_size: int = 64           # render font size before normalisation
    target_body: float = 20.0     # median letter ink height in the input (px)
    line_scale_jitter: float = 0.07  # per-line size variation (±7%)
    min_words: int = 1
    max_words: int = 4
    min_word_len: int = 2
    max_word_len: int = 12
    p_real_word: float = 0.70     # probability of a real word vs. pseudo word
    p_digits: float = 0.06        # probability of a numeric token
    p_light_aug: float = 1.0      # probability of applying image augmentations at all
    p_heavy_aug: float = 0.35     # probability of the "photo/camera" pipeline

    # ---- model ----------------------------------------------------------
    cnn_channels: tuple[int, ...] = (32, 32, 64, 64, 128, 128, 128)
    # Width reduction of 4 (two pooling layers) is deliberate: this
    # handwriting font is narrow, so a reduction of 8 leaves only ~2 CTC frames
    # per letter, which is not enough for CTC (it needs 2 frames + a blank).
    pool_after: tuple[int, ...] = (0, 2)
    rnn_hidden: int = 192
    rnn_layers: int = 2
    dropout: float = 0.1

    # ---- training -------------------------------------------------------
    seed: int = 1234
    batch_size: int = 32
    lr: float = 1e-3
    min_lr: float = 1e-4
    # Three stages: clean renderings first (the model learns the letter shapes
    # fastest there), then mild augmentations, then photo-like ones.  The first
    # stage also uses short texts only - a CTC model fits short sequences much
    # faster, which is a standard curriculum trick.
    steps_plain: int = 250
    steps_light: int = 400        # stage 2: mild augmentations
    steps_heavy: int = 300        # stage 3: photo-like augmentations
    steps_plain_max_words: int = 2  # length curriculum for the first stage
    warmup_steps: int = 50
    val_every: int = 100
    val_samples: int = 200
    log_every: int = 25
    grad_clip: float = 5.0
    workers: int = 3
    torch_threads: int = 4
    overfit_samples: int = 64     # used by the self-test sanity check

    # ---- fonts ----------------------------------------------------------
    font_dirs: list[str] = field(default_factory=lambda: [str(FONT_DIR)])
    # file name -> sampling weight (missing entries default to 1.0)
    font_weights: dict[str, float] = field(default_factory=dict)
    system_font_dirs: list[str] = field(
        default_factory=lambda: ["/usr/share/fonts/truetype/noto"]
    )

    # ---- paths ----------------------------------------------------------
    model_dir: str = str(MODEL_DIR)
    data_dir: str = str(DATA_DIR)
    runs_dir: str = str(RUNS_DIR)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in known:
                continue
            if key in {"cnn_channels", "pool_after"} and value is not None:
                value = tuple(value)
            kwargs[key] = value
        return cls(**kwargs)

    @property
    def width_reduction(self) -> int:
        """How much the CNN shrinks the width of the input."""
        return 2 ** len(self.pool_after)

    @property
    def max_label_len(self) -> int:
        """Longest CTC label that can still be decoded from ``max_width``."""
        return max(1, (self.max_width // self.width_reduction - 1) // 2)

"""Synthetic datasets.

Training data is generated on the fly: a map-style ``Dataset`` whose content
is a pure function of ``(seed, index)``.  This gives an effectively infinite
training stream while keeping the validation subset bit-for-bit reproducible,
and no disk space is required.

:class:`OnDiskDataset` reads an exported dataset (see :mod:`hwocr.generate`),
which is what the evaluation and the Tesseract baseline use.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from . import augment, bidi, charset, fonts, lexicon, render
from .config import Config

VAL_SEED = 424242
TEST_SEED = 7331733


class SyntheticLineDataset(Dataset):
    """Random text lines rendered with the registered fonts."""

    def __init__(
        self,
        cfg: Config,
        specs: Sequence[fonts.FontSpec] | None = None,
        weights: np.ndarray | None = None,
        seed: int = 0,
        length: int = 1_000_000,
        heavy_prob: float | None = None,
        augment_prob: float | None = None,
        words: Sequence[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.specs = list(specs) if specs else fonts.resolve_fonts(cfg)
        self.weights = (
            weights if weights is not None
            else fonts.sampling_weights(self.specs, cfg.font_weights)
        )
        self.seed = int(seed)
        self.length = int(length)
        self.heavy_prob = cfg.p_heavy_aug if heavy_prob is None else float(heavy_prob)
        # Probability that *any* image level augmentation is applied at all.
        # The first training stage runs with 0.0: a CRNN trained from scratch
        # converges much faster on clean renderings, the augmentations are then
        # faded in.
        self.augment_prob = (
            cfg.p_light_aug if augment_prob is None else float(augment_prob)
        )
        # Overridable per training stage (length curriculum).
        self.max_words = cfg.max_words
        self.max_word_len = cfg.max_word_len

        base_words = list(words) if words else lexicon.default_lexicon()
        self.prepared: list[tuple[fonts.FontSpec, set[str], list[str]]] = []
        for spec in self.specs:
            allowed = set(spec.supported())
            self.prepared.append((spec, allowed, lexicon.filter_lexicon(base_words, allowed)))

    def __len__(self) -> int:
        return self.length

    def sample_text(self, rng: random.Random) -> tuple[str, str, "fonts.FontSpec"]:
        """Returns (logical text, visual text, font)."""
        idx = int(rng.choices(range(len(self.prepared)), weights=self.weights, k=1)[0])
        spec, allowed, words = self.prepared[idx]
        text = lexicon.sample_text(rng, self.cfg, allowed, words,
                                   max_words=self.max_words,
                                   max_word_len=self.max_word_len)
        return text, bidi.logical_to_visual(text), spec

    def length_key(self, index: int) -> int:
        """Cheap estimate of the image width - used to bucket samples into batches.

        Generating only the text (no rendering, no augmentation) costs a
        fraction of a millisecond, and grouping similar lengths together cuts
        the amount of white padding in a batch dramatically.
        """
        rng = random.Random(self.seed * 1_000_003 + int(index))
        _, visual, _ = self.sample_text(rng)
        return len(visual)


    def render_sample(self, index: int) -> tuple[np.ndarray, str, str, "fonts.FontSpec"]:
        """Renders one sample: (uint8 image, logical text, visual text, font).

        Used both by ``__getitem__`` and by :mod:`hwocr.generate`, so an
        exported split is *exactly* the same data (same seed, same per-index
        random stream) as what the training loop validates on.
        """
        rng = random.Random(self.seed * 1_000_003 + int(index))
        _, visual, spec = self.sample_text(rng)
        image = render.render_line(visual, spec, rng, self.cfg)
        if self.augment_prob >= 1.0 or rng.random() < self.augment_prob:
            image = augment.apply_augment(image, rng, heavy=rng.random() < self.heavy_prob)
        return image, bidi.visual_to_logical(visual), visual, spec

    def __getitem__(self, index: int) -> dict:
        image, logical, visual, spec = self.render_sample(index)
        label = charset.encode(visual)
        if not label:  # pathological case - fall back to a fixed sample
            visual, label, logical = "םולש", charset.encode("םולש"), "שלום"
        arr, valid_w = augment.to_model_input(image, self.cfg.img_height, self.cfg.max_width)
        return {
            "image": torch.from_numpy(arr),
            "label": torch.tensor(label, dtype=torch.long),
            "valid_width": int(valid_w),
            "font": spec.name,
            "logical_text": logical,
        }


class OnDiskDataset(Dataset):
    """Dataset exported by :mod:`hwocr.generate` (``labels.tsv`` + PNG files)."""

    def __init__(self, root: str | Path, cfg: Config) -> None:
        self.root = Path(root)
        self.cfg = cfg
        self.entries: list[tuple[Path, str, str]] = []
        labels = self.root / "labels.tsv"
        if not labels.exists():
            raise FileNotFoundError(f"{labels} not found - run the 'generate' command first")
        for line in labels.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            self.entries.append((self.root / parts[0], parts[1], parts[2]))

    def length_key(self, index: int) -> int:
        return int(self.entries[index][1].__len__())

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        path, visual, logical = self.entries[index]
        image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        arr, valid_w = augment.to_model_input(image, self.cfg.img_height, self.cfg.max_width)
        return {
            "image": torch.from_numpy(arr),
            "label": torch.tensor(charset.encode(visual), dtype=torch.long),
            "valid_width": int(valid_w),
            "font": path.parent.name,
            "logical_text": logical,
            "path": str(path),
        }


class RealLineDataset(OnDiskDataset):
    """Hand-labelled crops with augmentation and repetition (fine-tuning).

    A handful of hand-labelled boxes is far too little to fine-tune on, so each
    label is repeated ``repeats`` times with an independent augmentation:
    20 labels can thus give a few hundred slightly different samples.
    ``heavy_prob`` decides how often the photo-like pipeline is used.
    """

    def __init__(
        self,
        root: str | Path,
        cfg: Config,
        repeats: int = 1,
        augment_prob: float = 0.7,
        heavy_prob: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__(root, cfg)
        self.repeats = max(1, int(repeats))
        self.augment_prob = float(augment_prob)
        self.heavy_prob = float(heavy_prob)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.entries) * self.repeats

    def length_key(self, index: int) -> int:
        # the index space is expanded by `repeats`, the key only depends on the base entry
        return super().length_key(index % len(self.entries))

    def __getitem__(self, index: int) -> dict:
        base = index % len(self.entries)
        variant = index // len(self.entries)
        path, visual, logical = self.entries[base]
        image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        if variant > 0 and self.augment_prob > 0:
            rng = random.Random(self.seed * 1_000_003 + base * 1009 + variant)
            if rng.random() < self.augment_prob:
                image = augment.apply_augment(image, rng,
                                              heavy=rng.random() < self.heavy_prob)
        arr, valid_w = augment.to_model_input(image, self.cfg.img_height, self.cfg.max_width)
        return {
            "image": torch.from_numpy(arr),
            "label": torch.tensor(charset.encode(visual), dtype=torch.long),
            "valid_width": int(valid_w),
            "font": "real",
            "logical_text": logical,
            "path": str(path),
        }


def collate(batch: list[dict], width_reduction: int = 8) -> dict:
    """Pad images/labels of a batch; keeps per-sample CTC lengths."""
    max_w = max(item["image"].shape[-1] for item in batch)
    max_w = int(np.ceil(max_w / width_reduction) * width_reduction)
    max_len = max(item["label"].numel() for item in batch)

    images = torch.full((len(batch), 1, batch[0]["image"].shape[1], max_w), 1.0)
    labels = torch.zeros((len(batch), max_len), dtype=torch.long)
    valid_widths = torch.zeros(len(batch), dtype=torch.long)
    label_lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        w = item["image"].shape[-1]
        images[i, :, :, :w] = item["image"]
        labels[i, :item["label"].numel()] = item["label"]
        valid_widths[i] = item["valid_width"]
        label_lengths[i] = item["label"].numel()
    return {
        "image": images,
        "label": labels,
        "valid_width": valid_widths,
        "label_length": label_lengths,
        "fonts": [item["font"] for item in batch],
        "logical_text": [item["logical_text"] for item in batch],
        "paths": [item.get("path", "") for item in batch],
    }


def ctc_input_lengths(valid_widths: torch.Tensor, width_reduction: int, max_t: int) -> torch.Tensor:
    """Convert pixel widths into CTC time steps (>= 1)."""
    lengths = torch.div(valid_widths, width_reduction, rounding_mode="floor")
    return lengths.clamp(min=1, max=max_t).to(torch.long)


class MixedLineDataset(Dataset):
    """Mixes the synthetic stream with hand-labelled real images.

    ``real`` is an :class:`OnDiskDataset` (a folder with ``labels.tsv``) - for
    example crops taken from the user's own documents and labelled by hand
    through the web UI.  A share of real samples adapts the recogniser to fonts
    or handwriting that a synthetic-only pipeline cannot know.

    The stage settings of the synthetic part (``heavy_prob``, ``augment_prob``,
    ``max_words``, ``max_word_len``) are forwarded to it, so the training loop
    keeps working unchanged.
    """

    def __init__(
        self,
        cfg: Config,
        synthetic: SyntheticLineDataset,
        real: "OnDiskDataset",
        p_real: float = 0.35,
        seed: int = 0,
        length: int = 1_000_000,
    ) -> None:
        self.cfg = cfg
        self.synthetic = synthetic
        self.real = real
        self.p_real = float(min(max(p_real, 0.0), 0.95))
        self.seed = int(seed)
        self.length = int(length)

    # ---- stage settings are forwarded to the synthetic part -------------
    @property
    def heavy_prob(self) -> float:
        return self.synthetic.heavy_prob

    @heavy_prob.setter
    def heavy_prob(self, value: float) -> None:
        self.synthetic.heavy_prob = value

    @property
    def augment_prob(self) -> float:
        return self.synthetic.augment_prob

    @augment_prob.setter
    def augment_prob(self, value: float) -> None:
        self.synthetic.augment_prob = value

    @property
    def max_words(self) -> int:
        return self.synthetic.max_words

    @max_words.setter
    def max_words(self, value: int) -> None:
        self.synthetic.max_words = value

    @property
    def max_word_len(self) -> int:
        return self.synthetic.max_word_len

    @max_word_len.setter
    def max_word_len(self, value: int) -> None:
        self.synthetic.max_word_len = value

    # ---- sampling --------------------------------------------------------
    def _pick(self, index: int) -> tuple[str, int]:
        rng = random.Random(self.seed * 1_000_003 + int(index))
        if rng.random() < self.p_real:
            return "real", rng.randrange(len(self.real))
        return "synthetic", rng.randrange(len(self.synthetic))

    def __len__(self) -> int:
        return self.length

    def length_key(self, index: int) -> int:
        source, sub = self._pick(index)
        if source == "real":
            return self.real.length_key(sub)
        return self.synthetic.length_key(sub)

    def __getitem__(self, index: int) -> dict:
        source, sub = self._pick(index)
        if source == "real":
            return self.real[sub]
        return self.synthetic[sub]


def worker_init(worker_id: int) -> None:
    # Data generation is CPU heavy; keep OpenCV single threaded inside workers.
    cv2.setNumThreads(0)
    np.random.seed(torch.initial_seed() % (2 ** 32))


_worker_init = worker_init  # backwards compatible alias


class BucketedBatchSampler(Sampler[list[int]]):
    """Batches of similar text length, to minimise padding inside a batch.

    A random window of indices is drawn, sorted by the (cheap) length key and
    cut into batches; the batches themselves are then shuffled.  On the
    synthetic dataset this removes roughly half of the wasted computation
    compared to completely random batches.
    """

    def __init__(
        self,
        key_fn: Callable[[int], int],
        total: int,
        batch_size: int,
        seed: int = 0,
        window: int | None = None,
        indices: Sequence[int] | None = None,
        shuffle: bool = True,
    ) -> None:
        self.key_fn = key_fn
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.shuffle = shuffle
        self.indices = list(indices) if indices is not None else None
        self.total = len(self.indices) if self.indices is not None else int(total)
        self.window = int(window or self.batch_size * 16)

    def __len__(self) -> int:
        windows = max(1, int(np.ceil(self.total / self.window)))
        return windows * max(1, self.window // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed * 7919 + self.epoch)
        self.epoch += 1
        remaining = self.total
        while remaining > 0:
            size = min(self.window, remaining)
            remaining -= size
            if self.indices is not None:
                start = self.total - remaining - size
                window = self.indices[start:start + size]
            else:
                window = [rng.randrange(self.total) for _ in range(size)]
            if self.shuffle:
                rng.shuffle(window)
            keyed = sorted(window, key=self.key_fn)
            batches = [keyed[i:i + self.batch_size] for i in range(0, len(keyed), self.batch_size)]
            if self.shuffle:
                batches = [b for b in batches if len(b) == self.batch_size]  # keep batches uniform
                rng.shuffle(batches)
            else:
                # evaluation: keep the (smaller) last batch so that every sample is scored
                batches = [b for b in batches if b]
            yield from batches


def make_loader(
    dataset: Dataset,
    cfg: Config,
    shuffle: bool,
    workers: int | None = None,
    bucketed: bool = True,
) -> DataLoader:
    """DataLoader with the project's collate/bucketing defaults."""
    workers = cfg.workers if workers is None else workers
    common = {
        "num_workers": workers,
        "collate_fn": lambda batch: collate(batch, cfg.width_reduction),
        "worker_init_fn": _worker_init,
    }
    if bucketed and hasattr(dataset, "length_key"):
        sampler = BucketedBatchSampler(
            key_fn=dataset.length_key,
            total=len(dataset),
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            shuffle=shuffle,
            indices=None if shuffle else range(len(dataset)),
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        **common,
    )



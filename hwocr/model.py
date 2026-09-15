"""CRNN + CTC recogniser.

``CNN`` (7 convolutions, width reduced 8x) -> ``BiLSTM`` -> per-frame
character log-probabilities, trained with the CTC loss.  Keeping the model
small on purpose: the proof of concept trains on a CPU in ~10 minutes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from . import charset
from .config import Config

CHECKPOINT_VERSION = 1


class CRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: Sequence[int] = (32, 32, 64, 64, 128, 128, 128),
        pool_after: Sequence[int] = (0, 2, 4),
        rnn_hidden: int = 192,
        rnn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.pool_after = tuple(pool_after)

        layers: list[nn.Module] = []
        prev = 1
        for i, ch in enumerate(channels):
            layers += [
                nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            if i in self.pool_after:
                layers.append(nn.MaxPool2d(2, 2))
            prev = ch
        self.cnn = nn.Sequential(*layers)
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=prev,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * rnn_hidden, self.num_classes)

    @property
    def width_reduction(self) -> int:
        return 2 ** len(self.pool_after)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, 1, H, W) -> log-probabilities (T, B, num_classes)."""
        features = self.cnn(x)
        features = self.height_pool(features).squeeze(2)      # (B, C, T)
        features = features.permute(0, 2, 1)                  # (B, T, C)
        out, _ = self.rnn(features)
        logits = self.fc(self.dropout(out))                   # (B, T, num_classes)
        return logits.log_softmax(dim=2).permute(1, 0, 2).contiguous()

    def feature_lengths(self, width: int) -> int:
        return max(1, int(width) // self.width_reduction)


def build_model(cfg: Config, num_classes: int | None = None) -> CRNN:
    return CRNN(
        num_classes=num_classes or charset.num_classes(),
        channels=cfg.cnn_channels,
        pool_after=cfg.pool_after,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        dropout=cfg.dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: str | Path,
    model: CRNN,
    cfg: Config,
    fonts_info: str = "",
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "config": cfg.to_dict(),
            "charset": charset.CHARS,
            "num_classes": model.num_classes,
            "model": model.state_dict(),
            "fonts_info": fonts_info,
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[CRNN, Config, dict[str, Any]]:
    """Load a trained model; returns (model, config, meta)."""
    blob = torch.load(str(path), map_location=device, weights_only=False)
    cfg = Config.from_dict(blob.get("config", {}))
    saved_charset = blob.get("charset", charset.CHARS)
    if saved_charset != charset.CHARS:
        raise ValueError(
            "Checkpoint was trained with a different alphabet - retrain "
            f"the model. Checkpoint: {saved_charset!r}"
        )
    model = CRNN(
        num_classes=int(blob.get("num_classes", charset.num_classes())),
        channels=cfg.cnn_channels,
        pool_after=cfg.pool_after,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        dropout=cfg.dropout,
    )
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    meta = {k: v for k, v in blob.items() if k not in {"model"}}
    return model, cfg, meta

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class AttackRange:
    low: float
    high: float

    @classmethod
    def from_config(cls, value: list[float] | tuple[float, float], default: tuple[float, float]) -> "AttackRange":
        if value is None:
            value = default
        return cls(float(value[0]), float(value[1]))

    def sample(self) -> float:
        return random.uniform(self.low, self.high)


class DifferentiableNoiseAttackLayer(nn.Module):
    """Noise-domain attacks used as a cheap training approximation.

    Running Stable Diffusion generation plus DDIM inversion inside every train
    step is too expensive and introduces non-differentiable image operations.
    These latent-noise attacks approximate the main corruption modes so the
    encoder/decoder can be trained efficiently. The real diffusion chain is
    evaluated separately in experiments/full_diffusion_eval.py.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        probabilities = dict(config.get("probabilities", {}))
        if not probabilities:
            probabilities = {"identity": 1.0}
        self.attack_names = list(probabilities.keys())
        probs = torch.tensor([float(probabilities[name]) for name in self.attack_names], dtype=torch.float32)
        if probs.sum() <= 0:
            self.attack_names = ["identity"]
            probs = torch.ones(1, dtype=torch.float32)
        self.register_buffer("probabilities", probs / probs.sum(), persistent=False)

        self.gaussian_lambda_range = AttackRange.from_config(
            config.get("gaussian_lambda_range"), (0.02, 0.30)
        )
        self.normalized_gaussian_lambda_range = AttackRange.from_config(
            config.get("normalized_gaussian_lambda_range", config.get("gaussian_lambda_range")),
            (0.02, 0.30),
        )
        self.mask_ratio_range = AttackRange.from_config(config.get("mask_ratio_range"), (0.05, 0.35))
        self.drop_ratio_range = AttackRange.from_config(config.get("drop_ratio_range"), (0.02, 0.25))
        self.channel_drop_count_range = AttackRange.from_config(
            config.get("channel_drop_count_range"), (1, 2)
        )
        self.affine_scale_range = AttackRange.from_config(config.get("affine_scale_range"), (0.90, 1.10))
        self.affine_shift_range = AttackRange.from_config(config.get("affine_shift_range"), (-0.10, 0.10))
        self.bias_range = AttackRange.from_config(config.get("bias_range"), (-0.15, 0.15))

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        batch = noise.shape[0]
        attack_indices = torch.multinomial(self.probabilities, batch, replacement=True)
        attacked = torch.empty_like(noise)
        for idx, name in enumerate(self.attack_names):
            selected = attack_indices == idx
            if selected.any():
                attacked[selected] = self.apply_named(name, noise[selected])
        return attacked

    def apply_named(self, name: str, noise: torch.Tensor, strength: float | None = None, **kwargs: Any) -> torch.Tensor:
        if name in ("identity", "clean", "none"):
            return noise
        if name == "gaussian":
            lam = self.gaussian_lambda_range.sample() if strength is None else float(strength)
            return self.simple_gaussian(noise, lam)
        if name == "normalized_gaussian":
            lam = self.normalized_gaussian_lambda_range.sample() if strength is None else float(strength)
            return self.normalized_gaussian(noise, lam)
        if name == "mask":
            ratio = self.mask_ratio_range.sample() if strength is None else float(strength)
            return self.mask_attack(noise, ratio)
        if name == "random_drop":
            ratio = self.drop_ratio_range.sample() if strength is None else float(strength)
            return self.random_drop(noise, ratio)
        if name == "channel_dropout":
            drop_count = self.channel_drop_count_range.sample() if strength is None else float(strength)
            return self.channel_dropout(noise, drop_count)
        if name == "affine":
            scale = kwargs.get("scale", self.affine_scale_range.sample())
            shift = kwargs.get("shift", self.affine_shift_range.sample())
            return self.affine_shift(noise, float(scale), float(shift))
        if name == "bias":
            bias = kwargs.get("bias", self.bias_range.sample() if strength is None else float(strength))
            return self.additive_bias(noise, float(bias))
        raise ValueError(f"Unknown noise attack: {name}")

    @staticmethod
    def simple_gaussian(noise: torch.Tensor, lam: float) -> torch.Tensor:
        if lam <= 0:
            return noise
        return noise + float(lam) * torch.randn_like(noise)

    @staticmethod
    def normalized_gaussian(noise: torch.Tensor, lam: float) -> torch.Tensor:
        if lam <= 0:
            return noise
        lam = float(lam)
        return (noise + lam * torch.randn_like(noise)) / math.sqrt(1.0 + lam * lam)

    @staticmethod
    def mask_attack(noise: torch.Tensor, ratio: float) -> torch.Tensor:
        ratio = max(0.0, min(float(ratio), 1.0))
        if ratio <= 0:
            return noise
        batch, _, height, width = noise.shape
        mask = torch.ones((batch, 1, height, width), device=noise.device, dtype=noise.dtype)
        patch_area = max(1, int(height * width * ratio))
        side = max(1, int(math.sqrt(patch_area)))
        patch_h = min(height, side)
        patch_w = min(width, max(1, patch_area // patch_h))
        for b in range(batch):
            top = int(torch.randint(0, height - patch_h + 1, (1,), device=noise.device).item())
            left = int(torch.randint(0, width - patch_w + 1, (1,), device=noise.device).item())
            mask[b, :, top : top + patch_h, left : left + patch_w] = 0
        return noise * mask

    @staticmethod
    def random_drop(noise: torch.Tensor, drop_probability: float) -> torch.Tensor:
        p = max(0.0, min(float(drop_probability), 1.0))
        if p <= 0:
            return noise
        keep = (torch.rand_like(noise) > p).to(noise.dtype)
        return noise * keep

    @staticmethod
    def channel_dropout(noise: torch.Tensor, drop_count: float) -> torch.Tensor:
        channels = noise.shape[1]
        count = int(round(float(drop_count)))
        count = max(0, min(count, channels - 1))
        if count <= 0:
            return noise
        keep = torch.ones(noise.shape[0], channels, 1, 1, device=noise.device, dtype=noise.dtype)
        scores = torch.rand(noise.shape[0], channels, device=noise.device)
        drop_indices = scores.argsort(dim=1)[:, :count]
        keep.scatter_(1, drop_indices[:, :, None, None], 0.0)
        return noise * keep

    @staticmethod
    def affine_shift(noise: torch.Tensor, scale: float, shift: float) -> torch.Tensor:
        return noise * float(scale) + float(shift)

    @staticmethod
    def additive_bias(noise: torch.Tensor, bias: float) -> torch.Tensor:
        return noise + float(bias)

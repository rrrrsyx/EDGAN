from __future__ import annotations

import math

import torch


def _limited_batch(noise: torch.Tensor, max_samples: int | None) -> torch.Tensor:
    x = noise.detach().float()
    if max_samples is not None and int(max_samples) > 0 and x.shape[0] > int(max_samples):
        x = x[: int(max_samples)]
    return x


def _spatial_corr_abs(
    x: torch.Tensor,
    shifts: tuple[tuple[int, int], ...],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    centered = x - x.mean(dim=(-2, -1), keepdim=True)
    denom = centered.pow(2).mean(dim=(-2, -1)).clamp_min(float(eps))
    values = []
    for shift in shifts:
        rolled = torch.roll(centered, shifts=shift, dims=(-2, -1))
        values.append((centered * rolled).mean(dim=(-2, -1)) / denom)
    corr = torch.stack(values, dim=0).abs()
    return corr.mean(), corr.max()


def _channel_corr_abs(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    channels = x.shape[1]
    if channels < 2:
        zero = x.new_zeros(())
        return zero, zero
    flattened = x.permute(1, 0, 2, 3).reshape(channels, -1)
    centered = flattened - flattened.mean(dim=1, keepdim=True)
    denom = centered.pow(2).mean(dim=1).clamp_min(float(eps)).sqrt()
    corr = (centered @ centered.t()) / max(centered.shape[1], 1)
    corr = corr / (denom[:, None] * denom[None, :]).clamp_min(float(eps))
    off_diag = ~torch.eye(channels, device=x.device, dtype=torch.bool)
    values = corr[off_diag].abs()
    return values.mean(), values.max()


def _fft_metrics(x: torch.Tensor, eps: float) -> dict[str, torch.Tensor]:
    fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    power = fft.abs().pow(2).flatten(start_dim=2)
    if power.shape[-1] > 1:
        power = power[..., 1:]
    if power.shape[-1] == 0:
        zero = x.new_zeros(())
        return {
            "rand_fft_entropy_norm": zero,
            "rand_fft_flatness": zero,
            "rand_fft_peak_ratio": zero,
            "rand_fft_valley_ratio": zero,
        }

    prob = power / power.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    entropy = -(prob * (prob + float(eps)).log()).sum(dim=-1)
    entropy_norm = entropy.mean() / math.log(float(power.shape[-1]))

    flatness = (power + float(eps)).log().mean(dim=-1).exp()
    flatness = flatness / power.mean(dim=-1).clamp_min(float(eps))

    avg_power = power.mean(dim=(0, 1))
    ratio = avg_power / avg_power.mean().clamp_min(float(eps))
    return {
        "rand_fft_entropy_norm": entropy_norm,
        "rand_fft_flatness": flatness.mean(),
        "rand_fft_peak_ratio": ratio.max(),
        "rand_fft_valley_ratio": ratio.min(),
    }


def gaussian_randomness_metrics(
    noise: torch.Tensor,
    max_samples: int | None = 128,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Evaluation-only diagnostics for N(0,1)-like white latent noise."""

    x = _limited_batch(noise, max_samples)
    flat = x.flatten()
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp_min(float(eps))
    z = (flat - mean) / std
    abs_mean = flat.abs().mean()
    positive_ratio = (flat > 0).float().mean()

    tail_prob_gt1 = (flat.abs() > 1.0).float().mean()
    tail_prob_gt2 = (flat.abs() > 2.0).float().mean()
    tail_prob_gt3 = (flat.abs() > 3.0).float().mean()
    tail_target_gt1 = x.new_tensor(math.erfc(1.0 / math.sqrt(2.0)))
    tail_target_gt2 = x.new_tensor(math.erfc(2.0 / math.sqrt(2.0)))
    tail_target_gt3 = x.new_tensor(math.erfc(3.0 / math.sqrt(2.0)))

    spatial_corr1_mean, spatial_corr1_max = _spatial_corr_abs(
        x,
        shifts=((0, 1), (1, 0), (1, 1), (1, -1)),
        eps=eps,
    )
    spatial_corr2_mean, spatial_corr2_max = _spatial_corr_abs(
        x,
        shifts=((0, 2), (2, 0), (2, 2), (2, -2)),
        eps=eps,
    )

    sign = torch.where(x >= 0, x.new_ones(()), -x.new_ones(()))
    sign_corr1_mean, sign_corr1_max = _spatial_corr_abs(
        sign,
        shifts=((0, 1), (1, 0), (1, 1), (1, -1)),
        eps=eps,
    )
    channel_corr_mean, channel_corr_max = _channel_corr_abs(x, eps=eps)

    position_mean = x.mean(dim=0)
    position_var = x.var(dim=0, unbiased=False) if x.shape[0] > 1 else x.new_zeros(x.shape[1:])

    metrics = {
        "rand_mean": mean,
        "rand_std": std,
        "rand_abs_mean": abs_mean,
        "rand_abs_mean_gap": (abs_mean - x.new_tensor(math.sqrt(2.0 / math.pi))).abs(),
        "rand_skew_abs": z.pow(3).mean().abs(),
        "rand_excess_kurtosis_abs": (z.pow(4).mean() - 3.0).abs(),
        "rand_abs_max": flat.abs().max(),
        "rand_positive_ratio": positive_ratio,
        "rand_sign_balance_abs": (positive_ratio - 0.5).abs(),
        "rand_tail_prob_gt1": tail_prob_gt1,
        "rand_tail_prob_gt2": tail_prob_gt2,
        "rand_tail_prob_gt3": tail_prob_gt3,
        "rand_tail_gap_gt1_abs": (tail_prob_gt1 - tail_target_gt1).abs(),
        "rand_tail_gap_gt2_abs": (tail_prob_gt2 - tail_target_gt2).abs(),
        "rand_tail_gap_gt3_abs": (tail_prob_gt3 - tail_target_gt3).abs(),
        "rand_spatial_corr1_abs_mean": spatial_corr1_mean,
        "rand_spatial_corr1_abs_max": spatial_corr1_max,
        "rand_spatial_corr2_abs_mean": spatial_corr2_mean,
        "rand_spatial_corr2_abs_max": spatial_corr2_max,
        "rand_sign_corr1_abs_mean": sign_corr1_mean,
        "rand_sign_corr1_abs_max": sign_corr1_max,
        "rand_channel_corr_abs_mean": channel_corr_mean,
        "rand_channel_corr_abs_max": channel_corr_max,
        "rand_position_mean_rms": position_mean.pow(2).mean().sqrt(),
        "rand_position_var_mean": position_var.mean(),
        "rand_position_var_std": position_var.std(unbiased=False),
    }
    metrics.update(_fft_metrics(x, eps=eps))
    return metrics

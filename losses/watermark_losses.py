from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from utils.metrics import bit_accuracy_from_logits


def standard_normal_pdf_loss(
    noise: torch.Tensor,
    sample_ratio: float = 0.25,
    num_samples: int | None = None,
    num_bins: int = 64,
    value_range: float = 4.0,
    hist_chunk_size: int = 128,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match sampled channel histograms to the N(0,1) PDF.

    Each [sample, channel] map samples a ratio of its own spatial values and
    builds a differentiable soft histogram with triangular bins. The target is
    the standard-normal probability mass for the same bin centers.
    """

    batch, channels, height, width = noise.shape
    x = noise.float().reshape(batch * channels, height * width)
    available = x.shape[1]
    if num_samples is None or int(num_samples) <= 0:
        samples = int(round(available * float(sample_ratio)))
    else:
        samples = int(num_samples)
    samples = min(max(samples, 0), available)
    bins = max(int(num_bins), 0)
    if samples <= 0 or bins <= 1:
        return noise.new_zeros(())

    indices = torch.randint(available, (x.shape[0], samples), device=noise.device)
    sampled = x.gather(dim=1, index=indices).clamp(-float(value_range), float(value_range))

    centers = torch.linspace(-float(value_range), float(value_range), bins, device=noise.device, dtype=x.dtype)
    bin_width = centers[1] - centers[0]
    target_pdf = torch.exp(-0.5 * centers.pow(2)) / torch.sqrt(
        torch.tensor(2.0 * torch.pi, device=noise.device, dtype=x.dtype)
    )
    target_mass = target_pdf * bin_width
    target_mass = target_mass / target_mass.sum().clamp_min(eps)

    chunk_size = max(int(hist_chunk_size), 1)
    loss_sum = noise.new_zeros(())
    item_count = 0
    for start in range(0, sampled.shape[0], chunk_size):
        chunk = sampled[start : start + chunk_size]
        distances = (chunk.unsqueeze(-1) - centers.view(1, 1, -1)).abs() / bin_width.clamp_min(eps)
        weights = (1.0 - distances).clamp_min(0.0)
        hist = weights.sum(dim=1)
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)
        loss_sum = loss_sum + F.mse_loss(hist, target_mass.unsqueeze(0).expand_as(hist), reduction="sum")
        item_count += hist.numel()
    return loss_sum / max(item_count, 1)


def gaussian_kl_loss(noise: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Moment KL from each empirical [sample, channel] map to N(0,1)."""

    batch, channels, _, _ = noise.shape
    x = noise.float().reshape(batch, channels, -1)
    mean = x.mean(dim=-1)
    var = x.var(dim=-1, unbiased=False).clamp_min(eps)
    kl = 0.5 * (mean.pow(2) + var - var.log() - 1.0)
    return kl.mean()


def tail_penalty_loss(noise: torch.Tensor, value_range: float = 4.0) -> torch.Tensor:
    """Penalize values outside the normal operating range without clipping them."""

    excess = F.relu(noise.float().abs() - float(value_range))
    return excess.pow(2).mean()


def frequency_entropy(noise: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = noise.float()
    fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    power = fft.abs().pow(2).flatten(start_dim=2)
    prob = power / power.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(prob * (prob + eps).log()).sum(dim=-1)
    return entropy.mean()


def frequency_entropy_loss(
    noise: torch.Tensor,
    target: float = 7.895,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entropy = frequency_entropy(noise, eps=eps)
    target_tensor = noise.new_tensor(float(target))
    gap = entropy - target_tensor
    return gap.pow(2), entropy, gap


def frequency_uniformity_loss(noise: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return frequency_entropy_loss(noise, eps=eps)[0]


def spectral_band_stats(
    noise: torch.Tensor,
    tolerance: float = 0.2,
    ignore_dc: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = noise.float()
    fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    power = fft.abs().pow(2).flatten(start_dim=2)
    if ignore_dc and power.shape[-1] > 1:
        power = power[..., 1:]

    avg_power = power.mean(dim=0)
    mean_power = avg_power.mean(dim=-1, keepdim=True).clamp_min(eps)
    ratio = (avg_power / mean_power).clamp_min(eps)
    log_ratio_abs = ratio.log().abs()
    excess = F.relu(log_ratio_abs - float(tolerance))
    loss = excess.pow(2).mean()
    return loss, log_ratio_abs.mean(), log_ratio_abs.max(), ratio.min(), ratio.max()


def spectral_band_loss(
    noise: torch.Tensor,
    tolerance: float = 0.2,
    ignore_dc: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    return spectral_band_stats(noise, tolerance=tolerance, ignore_dc=ignore_dc, eps=eps)[0]


def sign_spectrum_band_stats(
    noise: torch.Tensor,
    temperature: float = 0.5,
    lower: float | None = None,
    upper: float | None = None,
    tolerance: float | None = 0.15,
    topk: int = 32,
    topk_weight: float = 1.0,
    ignore_dc: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bound the soft-sign spectrum around the white-noise average line.

    The hard sign test `(noise > 0)` is not differentiable. `tanh(noise / T)`
    keeps the sign structure trainable while still emphasizing positive/negative
    placement rather than exact amplitudes.
    """

    if lower is None or upper is None:
        tol = max(float(tolerance if tolerance is not None else 0.15), 0.0)
        lower = math.exp(-tol)
        upper = math.exp(tol)
    lower = max(float(lower), float(eps))
    upper = max(float(upper), lower + float(eps))
    temp = max(float(temperature), float(eps))
    signs = torch.tanh(noise.float() / temp)
    fft = torch.fft.fft2(signs, dim=(-2, -1), norm="ortho")
    power = fft.abs().pow(2).flatten(start_dim=2)
    if ignore_dc and power.shape[-1] > 1:
        power = power[..., 1:]

    avg_power = power.mean(dim=0)
    mean_power = avg_power.mean(dim=-1, keepdim=True).clamp_min(eps)
    ratio = (avg_power / mean_power).clamp_min(eps)
    excess = F.relu(lower - ratio) + F.relu(ratio - upper)
    mean_loss = excess.pow(2).mean()
    k = min(max(int(topk), 1), excess.shape[-1])
    topk_loss = excess.topk(k=k, dim=-1).values.pow(2).mean()
    loss = mean_loss + float(topk_weight) * topk_loss
    return loss, mean_loss, topk_loss, ratio.min(), ratio.max()


def sign_spectrum_band_loss(
    noise: torch.Tensor,
    temperature: float = 0.5,
    lower: float | None = None,
    upper: float | None = None,
    tolerance: float | None = 0.15,
    topk: int = 32,
    topk_weight: float = 1.0,
    ignore_dc: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    return sign_spectrum_band_stats(
        noise,
        temperature=temperature,
        lower=lower,
        upper=upper,
        tolerance=tolerance,
        topk=topk,
        topk_weight=topk_weight,
        ignore_dc=ignore_dc,
        eps=eps,
    )[0]


def channel_correlation_loss(noise: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable proxy for MI/channel-independence.

    Direct neural mutual-information estimation is often noisy and expensive in
    this setting. Penalizing off-diagonal channel correlations is a deterministic
    second-order approximation: independent channels should have near-zero
    correlation, which lowers redundant information sharing across channels.
    """

    batch, channels, height, width = noise.shape
    x = noise.reshape(batch, channels, height * width).float()
    x = x - x.mean(dim=2, keepdim=True)
    x = x / x.std(dim=2, keepdim=True, unbiased=False).clamp_min(eps)
    corr = torch.bmm(x, x.transpose(1, 2)) / x.shape[2]
    eye = torch.eye(channels, device=noise.device, dtype=torch.bool)
    return corr[:, ~eye].pow(2).mean()


def same_position_channel_corr_stats(
    noise: torch.Tensor,
    tolerance: float = 0.08,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize channel dependence at the same spatial coordinate across a batch."""

    batch, channels, height, width = noise.shape
    if batch <= 1 or channels <= 1:
        zero = noise.new_zeros(())
        return zero, zero, zero

    x = noise.float().permute(2, 3, 1, 0).reshape(height * width, channels, batch)
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    corr = torch.bmm(x, x.transpose(1, 2)) / float(batch)
    eye = torch.eye(channels, device=noise.device, dtype=torch.bool)
    off_diag = corr[:, ~eye]
    abs_corr = off_diag.abs()
    loss = F.relu(abs_corr - float(tolerance)).pow(2).mean()
    return loss, abs_corr.mean(), abs_corr.max()


def same_position_channel_corr_loss(
    noise: torch.Tensor,
    tolerance: float = 0.08,
    eps: float = 1e-6,
) -> torch.Tensor:
    return same_position_channel_corr_stats(noise, tolerance=tolerance, eps=eps)[0]


def local_mean_loss(
    noise: torch.Tensor,
    patch_size: int = 8,
    num_patches: int = 8,
    target: float | None = None,
) -> torch.Tensor:
    stat = local_mean_stat(noise, patch_size=patch_size, num_patches=num_patches)
    if target is None:
        target = 1.0 / float(min(patch_size, noise.shape[2]) * min(patch_size, noise.shape[3]))
    return (stat - float(target)).pow(2)


def local_mean_stat(
    noise: torch.Tensor,
    patch_size: int = 8,
    num_patches: int = 8,
) -> torch.Tensor:
    if num_patches <= 0:
        return noise.new_zeros(())
    _, _, height, width = noise.shape
    patch_h = min(patch_size, height)
    patch_w = min(patch_size, width)
    max_y = height - patch_h
    max_x = width - patch_w
    losses = []
    for _ in range(num_patches):
        top = int(torch.randint(0, max_y + 1, (1,), device=noise.device).item())
        left = int(torch.randint(0, max_x + 1, (1,), device=noise.device).item())
        patch = noise[:, :, top : top + patch_h, left : left + patch_w]
        losses.append(patch.mean(dim=(2, 3)).pow(2).mean())
    return torch.stack(losses).mean()


def batch_position_kl_stats(
    noise: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = noise.float()
    if x.shape[0] <= 1:
        zero = x.new_zeros(())
        return zero, zero, zero, zero, zero, zero, zero

    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False).clamp_min(eps)
    mean_term = 0.5 * mean.pow(2).mean()
    var_term = 0.5 * (var - var.log() - 1.0).mean()
    loss = mean_term + var_term
    mean_energy = mean.pow(2).mean()
    mean_rms = mean_energy.sqrt()
    var_mean = var.mean()
    var_spatial_std = var.std(unbiased=False)
    return loss, mean_term, var_term, mean_rms, mean_energy, var_mean, var_spatial_std


def batch_position_kl_loss(noise: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return batch_position_kl_stats(noise, eps=eps)[0]


def fft_autocorr_stats(
    noise: torch.Tensor,
    topk: int = 32,
    topk_weight: float = 0.5,
    peak_target: float = 0.08,
    peak_weight: float = 0.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full circular spatial autocorrelation loss for every sample/channel."""

    x = noise.float()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fft2(x, dim=(-2, -1))
    autocorr = torch.fft.ifft2(fft.abs().pow(2), dim=(-2, -1)).real
    autocorr = autocorr / autocorr[..., :1, :1].clamp_min(float(eps))
    flat = autocorr.flatten(start_dim=2)
    off_center = flat[..., 1:]
    abs_corr = off_center.abs()
    mean_loss = off_center.pow(2).mean()
    k = min(max(int(topk), 1), abs_corr.shape[-1])
    topk_values = abs_corr.topk(k=k, dim=-1).values
    topk_loss = topk_values.pow(2).mean()
    per_map_peak = abs_corr.max(dim=-1).values
    peak_loss = F.relu(per_map_peak - float(peak_target)).pow(2).mean()
    peak_q95 = torch.quantile(per_map_peak.flatten(), 0.95)
    peak_q99 = torch.quantile(per_map_peak.flatten(), 0.99)
    loss = mean_loss + float(topk_weight) * topk_loss + float(peak_weight) * peak_loss
    return loss, mean_loss, topk_loss, peak_loss, per_map_peak.mean(), peak_q95, peak_q99, abs_corr.max()


def fft_autocorr_loss(
    noise: torch.Tensor,
    topk: int = 32,
    topk_weight: float = 0.5,
    peak_target: float = 0.08,
    peak_weight: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    return fft_autocorr_stats(
        noise,
        topk=topk,
        topk_weight=topk_weight,
        peak_target=peak_target,
        peak_weight=peak_weight,
        eps=eps,
    )[0]


@dataclass
class LossWeights:
    lambda_bit: float = 1.0
    lambda_dist: float = 0.02
    lambda_kl: float = 0.0
    lambda_tail: float = 0.0
    lambda_freq: float = 0.0
    lambda_spectral_band: float = 0.0
    lambda_sign_spectrum: float = 0.0
    lambda_mi: float = 0.01
    lambda_same_position_channel_corr: float = 0.0
    lambda_mean: float = 0.01
    lambda_batch_position_kl: float = 0.0
    lambda_fft_autocorr: float = 0.0


def compute_watermark_losses(
    logits: torch.Tensor,
    bits: torch.Tensor,
    noise: torch.Tensor,
    weights: LossWeights | dict[str, float] | None = None,
    local_mean_patch_size: int = 8,
    local_mean_num_patches: int = 8,
    dist_sample_ratio: float = 0.25,
    dist_num_samples: int | None = None,
    dist_num_bins: int = 64,
    dist_value_range: float = 4.0,
    dist_hist_chunk_size: int = 128,
    tail_value_range: float = 4.0,
    freq_target: float = 7.895,
    spectral_band_tolerance: float = 0.2,
    spectral_band_ignore_dc: bool = True,
    sign_spectrum_temperature: float = 0.5,
    sign_spectrum_lower: float | None = None,
    sign_spectrum_upper: float | None = None,
    sign_spectrum_tolerance: float | None = 0.15,
    sign_spectrum_topk: int = 32,
    sign_spectrum_topk_weight: float = 1.0,
    sign_spectrum_ignore_dc: bool = True,
    same_position_channel_corr_tolerance: float = 0.08,
    same_position_channel_corr_eps: float = 1e-6,
    fft_autocorr_topk: int = 32,
    fft_autocorr_topk_weight: float = 0.5,
    fft_autocorr_peak_target: float = 0.08,
    fft_autocorr_peak_weight: float = 0.0,
    fft_autocorr_eps: float = 1e-6,
    local_mean_target: float | None = None,
    batch_position_kl_eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if weights is None:
        weights = LossWeights()
    elif isinstance(weights, dict):
        values = {k: float(v) for k, v in weights.items() if k in LossWeights.__annotations__}
        weights = LossWeights(**values)

    bits = bits.float()
    zero = noise.new_zeros(())
    loss_bit = F.binary_cross_entropy_with_logits(logits, bits)
    if weights.lambda_dist != 0.0:
        loss_dist = standard_normal_pdf_loss(
            noise,
            sample_ratio=dist_sample_ratio,
            num_samples=dist_num_samples,
            num_bins=dist_num_bins,
            value_range=dist_value_range,
            hist_chunk_size=dist_hist_chunk_size,
        )
    else:
        loss_dist = zero
    loss_kl = gaussian_kl_loss(noise) if weights.lambda_kl != 0.0 else zero
    loss_tail = tail_penalty_loss(noise, value_range=tail_value_range) if weights.lambda_tail != 0.0 else zero
    if weights.lambda_freq != 0.0:
        loss_freq, freq_entropy_value, _freq_entropy_gap = frequency_entropy_loss(noise, target=freq_target)
    else:
        loss_freq = zero
        freq_entropy_value = zero
    if weights.lambda_spectral_band != 0.0:
        (
            loss_spectral_band,
            _spectral_band_log_abs_mean,
            _spectral_band_log_abs_max,
            spectral_band_ratio_min,
            spectral_band_ratio_max,
        ) = spectral_band_stats(
            noise,
            tolerance=spectral_band_tolerance,
            ignore_dc=spectral_band_ignore_dc,
        )
    else:
        loss_spectral_band = zero
        spectral_band_ratio_min = zero
        spectral_band_ratio_max = zero
    if weights.lambda_sign_spectrum != 0.0:
        (
            loss_sign_spectrum,
            sign_spectrum_mean_loss,
            sign_spectrum_topk_loss,
            sign_spectrum_ratio_min,
            sign_spectrum_ratio_max,
        ) = sign_spectrum_band_stats(
            noise,
            temperature=sign_spectrum_temperature,
            lower=sign_spectrum_lower,
            upper=sign_spectrum_upper,
            tolerance=sign_spectrum_tolerance,
            topk=sign_spectrum_topk,
            topk_weight=sign_spectrum_topk_weight,
            ignore_dc=sign_spectrum_ignore_dc,
        )
    else:
        loss_sign_spectrum = zero
        sign_spectrum_mean_loss = zero
        sign_spectrum_topk_loss = zero
        sign_spectrum_ratio_min = zero
        sign_spectrum_ratio_max = zero
    loss_mi = channel_correlation_loss(noise) if weights.lambda_mi != 0.0 else zero
    if weights.lambda_same_position_channel_corr != 0.0:
        (
            loss_same_position_channel_corr,
            same_position_channel_corr_abs_mean,
            same_position_channel_corr_abs_max,
        ) = same_position_channel_corr_stats(
            noise,
            tolerance=same_position_channel_corr_tolerance,
            eps=same_position_channel_corr_eps,
        )
    else:
        loss_same_position_channel_corr = zero
        same_position_channel_corr_abs_mean = zero
        same_position_channel_corr_abs_max = zero
    if weights.lambda_batch_position_kl != 0.0:
        (
            loss_batch_position_kl,
            batch_position_kl_mean_term,
            batch_position_kl_var_term,
            batch_position_mean_rms,
            batch_position_mean_energy,
            batch_position_var_mean,
            batch_position_var_std,
        ) = batch_position_kl_stats(noise, eps=batch_position_kl_eps)
    else:
        loss_batch_position_kl = zero
        batch_position_kl_mean_term = zero
        batch_position_kl_var_term = zero
        batch_position_mean_rms = zero
        batch_position_mean_energy = zero
        batch_position_var_mean = zero
        batch_position_var_std = zero
    if weights.lambda_fft_autocorr != 0.0:
        (
            loss_fft_autocorr,
            fft_autocorr_mean_loss,
            fft_autocorr_topk_loss,
            fft_autocorr_peak_loss,
            fft_autocorr_peak_mean,
            fft_autocorr_peak_q95,
            fft_autocorr_peak_q99,
            fft_autocorr_abs_max,
        ) = fft_autocorr_stats(
            noise,
            topk=fft_autocorr_topk,
            topk_weight=fft_autocorr_topk_weight,
            peak_target=fft_autocorr_peak_target,
            peak_weight=fft_autocorr_peak_weight,
            eps=fft_autocorr_eps,
        )
    else:
        loss_fft_autocorr = zero
        fft_autocorr_mean_loss = zero
        fft_autocorr_topk_loss = zero
        fft_autocorr_peak_loss = zero
        fft_autocorr_peak_mean = zero
        fft_autocorr_peak_q95 = zero
        fft_autocorr_peak_q99 = zero
        fft_autocorr_abs_max = zero
    if weights.lambda_mean != 0.0:
        local_mean_value = local_mean_stat(
            noise,
            patch_size=local_mean_patch_size,
            num_patches=local_mean_num_patches,
        )
        if local_mean_target is None:
            patch_h = min(local_mean_patch_size, noise.shape[2])
            patch_w = min(local_mean_patch_size, noise.shape[3])
            local_mean_target = 1.0 / float(patch_h * patch_w)
        loss_mean = (local_mean_value - float(local_mean_target)).pow(2)
    else:
        local_mean_value = noise.new_zeros(())
        loss_mean = noise.new_zeros(())

    total = (
        weights.lambda_bit * loss_bit
        + weights.lambda_dist * loss_dist
        + weights.lambda_kl * loss_kl
        + weights.lambda_tail * loss_tail
        + weights.lambda_freq * loss_freq
        + weights.lambda_spectral_band * loss_spectral_band
        + weights.lambda_sign_spectrum * loss_sign_spectrum
        + weights.lambda_mi * loss_mi
        + weights.lambda_same_position_channel_corr * loss_same_position_channel_corr
        + weights.lambda_mean * loss_mean
        + weights.lambda_batch_position_kl * loss_batch_position_kl
        + weights.lambda_fft_autocorr * loss_fft_autocorr
    )
    metrics = {
        "loss_total": total.detach(),
        "loss_bit": loss_bit.detach(),
        "loss_dist": loss_dist.detach(),
        "loss_kl": loss_kl.detach(),
        "loss_tail": loss_tail.detach(),
        "loss_freq": loss_freq.detach(),
        "freq_entropy": freq_entropy_value.detach(),
        "loss_spectral_band": loss_spectral_band.detach(),
        "spectral_band_ratio_min": spectral_band_ratio_min.detach(),
        "spectral_band_ratio_max": spectral_band_ratio_max.detach(),
        "loss_sign_spectrum": loss_sign_spectrum.detach(),
        "sign_spectrum_mean_loss": sign_spectrum_mean_loss.detach(),
        "sign_spectrum_topk_loss": sign_spectrum_topk_loss.detach(),
        "sign_spectrum_ratio_min": sign_spectrum_ratio_min.detach(),
        "sign_spectrum_ratio_max": sign_spectrum_ratio_max.detach(),
        "loss_mi": loss_mi.detach(),
        "loss_same_position_channel_corr": loss_same_position_channel_corr.detach(),
        "same_position_channel_corr_abs_mean": same_position_channel_corr_abs_mean.detach(),
        "same_position_channel_corr_abs_max": same_position_channel_corr_abs_max.detach(),
        "loss_mean": loss_mean.detach(),
        "loss_batch_position_kl": loss_batch_position_kl.detach(),
        "batch_position_kl_mean_term": batch_position_kl_mean_term.detach(),
        "batch_position_kl_var_term": batch_position_kl_var_term.detach(),
        "batch_position_mean_rms": batch_position_mean_rms.detach(),
        "batch_position_mean_energy": batch_position_mean_energy.detach(),
        "batch_position_var_mean": batch_position_var_mean.detach(),
        "batch_position_var_std": batch_position_var_std.detach(),
        "loss_fft_autocorr": loss_fft_autocorr.detach(),
        "fft_autocorr_mean_loss": fft_autocorr_mean_loss.detach(),
        "fft_autocorr_topk_loss": fft_autocorr_topk_loss.detach(),
        "fft_autocorr_peak_loss": fft_autocorr_peak_loss.detach(),
        "fft_autocorr_peak_mean": fft_autocorr_peak_mean.detach(),
        "fft_autocorr_peak_q95": fft_autocorr_peak_q95.detach(),
        "fft_autocorr_peak_q99": fft_autocorr_peak_q99.detach(),
        "fft_autocorr_abs_max": fft_autocorr_abs_max.detach(),
        "local_mean_stat": local_mean_value.detach(),
    }
    metrics["watermark_bit_acc"] = bit_accuracy_from_logits(logits.detach(), bits.detach())
    metrics["bit_acc"] = metrics["watermark_bit_acc"]
    return total, metrics


class WatermarkLoss(nn.Module):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        self.weights = LossWeights(
            lambda_bit=float(config.get("lambda_bit", 1.0)),
            lambda_dist=float(config.get("lambda_dist", 0.02)),
            lambda_kl=float(config.get("lambda_kl", 0.0)),
            lambda_tail=float(config.get("lambda_tail", 0.0)),
            lambda_freq=float(config.get("lambda_freq", 0.0)),
            lambda_spectral_band=float(config.get("lambda_spectral_band", 0.0)),
            lambda_sign_spectrum=float(config.get("lambda_sign_spectrum", 0.0)),
            lambda_mi=float(config.get("lambda_mi", 0.01)),
            lambda_same_position_channel_corr=float(config.get("lambda_same_position_channel_corr", 0.0)),
            lambda_mean=float(config.get("lambda_mean", 0.01)),
            lambda_batch_position_kl=float(
                config.get(
                    "lambda_batch_position_kl",
                    config.get("lambda_batch_mean_template", config.get("lambda_template", 0.0)),
                )
            ),
            lambda_fft_autocorr=float(config.get("lambda_fft_autocorr", 0.0)),
        )
        self.local_mean_patch_size = int(config.get("local_mean_patch_size", 8))
        self.local_mean_num_patches = int(config.get("local_mean_num_patches", 8))
        self.dist_sample_ratio = float(config.get("dist_sample_ratio", 0.25))
        self.dist_num_samples = (
            int(config["dist_num_samples"]) if "dist_num_samples" in config and config["dist_num_samples"] else None
        )
        self.dist_num_bins = int(config.get("dist_num_bins", 64))
        self.dist_value_range = float(config.get("dist_value_range", 4.0))
        self.dist_hist_chunk_size = int(config.get("dist_hist_chunk_size", 128))
        self.tail_value_range = float(config.get("tail_value_range", self.dist_value_range))
        self.freq_target = float(config.get("freq_entropy_target", config.get("freq_target", 7.895)))
        self.spectral_band_tolerance = float(config.get("spectral_band_tolerance", 0.2))
        band_ignore_dc = config.get("spectral_band_ignore_dc", True)
        self.spectral_band_ignore_dc = (
            band_ignore_dc.strip().lower() not in ("0", "false", "no", "off")
            if isinstance(band_ignore_dc, str)
            else bool(band_ignore_dc)
        )
        self.sign_spectrum_temperature = float(config.get("sign_spectrum_temperature", 0.5))
        self.sign_spectrum_lower = (
            float(config["sign_spectrum_lower"])
            if "sign_spectrum_lower" in config and config["sign_spectrum_lower"] not in (None, "auto")
            else None
        )
        self.sign_spectrum_upper = (
            float(config["sign_spectrum_upper"])
            if "sign_spectrum_upper" in config and config["sign_spectrum_upper"] not in (None, "auto")
            else None
        )
        self.sign_spectrum_tolerance = (
            float(config["sign_spectrum_tolerance"])
            if "sign_spectrum_tolerance" in config and config["sign_spectrum_tolerance"] not in (None, "auto")
            else None
        )
        self.sign_spectrum_topk = int(config.get("sign_spectrum_topk", 32))
        self.sign_spectrum_topk_weight = float(config.get("sign_spectrum_topk_weight", 1.0))
        sign_ignore_dc = config.get("sign_spectrum_ignore_dc", True)
        self.sign_spectrum_ignore_dc = (
            sign_ignore_dc.strip().lower() not in ("0", "false", "no", "off")
            if isinstance(sign_ignore_dc, str)
            else bool(sign_ignore_dc)
        )
        self.same_position_channel_corr_tolerance = float(
            config.get("same_position_channel_corr_tolerance", 0.08)
        )
        self.same_position_channel_corr_eps = float(config.get("same_position_channel_corr_eps", 1e-6))
        self.fft_autocorr_topk = int(config.get("fft_autocorr_topk", 32))
        self.fft_autocorr_topk_weight = float(config.get("fft_autocorr_topk_weight", 0.5))
        self.fft_autocorr_peak_target = float(config.get("fft_autocorr_peak_target", 0.08))
        self.fft_autocorr_peak_weight = float(config.get("fft_autocorr_peak_weight", 0.0))
        self.fft_autocorr_eps = float(config.get("fft_autocorr_eps", 1e-6))
        self.local_mean_target = (
            float(config["local_mean_target"])
            if "local_mean_target" in config and config["local_mean_target"] not in (None, "auto")
            else None
        )
        self.batch_position_kl_eps = float(config.get("batch_position_kl_eps", 1e-6))

    def forward(
        self, logits: torch.Tensor, bits: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return compute_watermark_losses(
            logits=logits,
            bits=bits,
            noise=noise,
            weights=self.weights,
            local_mean_patch_size=self.local_mean_patch_size,
            local_mean_num_patches=self.local_mean_num_patches,
            dist_sample_ratio=self.dist_sample_ratio,
            dist_num_samples=self.dist_num_samples,
            dist_num_bins=self.dist_num_bins,
            dist_value_range=self.dist_value_range,
            dist_hist_chunk_size=self.dist_hist_chunk_size,
            tail_value_range=self.tail_value_range,
            freq_target=self.freq_target,
            spectral_band_tolerance=self.spectral_band_tolerance,
            spectral_band_ignore_dc=self.spectral_band_ignore_dc,
            sign_spectrum_temperature=self.sign_spectrum_temperature,
            sign_spectrum_lower=self.sign_spectrum_lower,
            sign_spectrum_upper=self.sign_spectrum_upper,
            sign_spectrum_tolerance=self.sign_spectrum_tolerance,
            sign_spectrum_topk=self.sign_spectrum_topk,
            sign_spectrum_topk_weight=self.sign_spectrum_topk_weight,
            sign_spectrum_ignore_dc=self.sign_spectrum_ignore_dc,
            same_position_channel_corr_tolerance=self.same_position_channel_corr_tolerance,
            same_position_channel_corr_eps=self.same_position_channel_corr_eps,
            fft_autocorr_topk=self.fft_autocorr_topk,
            fft_autocorr_topk_weight=self.fft_autocorr_topk_weight,
            fft_autocorr_peak_target=self.fft_autocorr_peak_target,
            fft_autocorr_peak_weight=self.fft_autocorr_peak_weight,
            fft_autocorr_eps=self.fft_autocorr_eps,
            local_mean_target=self.local_mean_target,
            batch_position_kl_eps=self.batch_position_kl_eps,
        )

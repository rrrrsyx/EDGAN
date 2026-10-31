from __future__ import annotations

from dataclasses import dataclass

import torch


def logits_to_bits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).to(torch.uint8)


def bit_accuracy_from_logits(
    logits: torch.Tensor, bits: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    pred = logits_to_bits(logits, threshold=threshold).bool()
    target = bits.bool()
    return (pred == target).float().mean()


def bit_accuracy(pred_bits: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
    return (pred_bits.bool() == target_bits.bool()).float().mean()


def per_sample_bit_accuracy(pred_bits: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
    return (pred_bits.bool() == target_bits.bool()).float().mean(dim=1)


@dataclass
class TraceabilityResult:
    trace_accuracy: float
    mean_top1_match: float
    mean_true_match: float
    pool_size: int


def traceability_top1(
    extracted_bits: torch.Tensor,
    true_bits: torch.Tensor,
    pool_size: int,
    chunk_size: int = 8192,
    device: str | torch.device = "cpu",
    seed: int = 1234,
) -> TraceabilityResult:
    """Top-1 watermark attribution against a large candidate pool.

    The first N candidates are overwritten with the true watermarks, so the
    expected true index for sample i is i. Similarity is computed with matrix
    multiplication on {-1, +1} bits, avoiding an N x pool x bit_dim tensor.
    """

    extracted_bits = extracted_bits.detach().to(torch.uint8).cpu()
    true_bits = true_bits.detach().to(torch.uint8).cpu()
    num_samples, bit_dim = true_bits.shape
    if extracted_bits.shape != true_bits.shape:
        raise ValueError("extracted_bits and true_bits must have the same shape")
    if pool_size < num_samples:
        raise ValueError("pool_size must be >= number of true samples")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidate_pool = torch.randint(
        0, 2, (pool_size, bit_dim), generator=generator, dtype=torch.uint8
    )
    candidate_pool[:num_samples] = true_bits

    device = torch.device(device)
    extracted_sign = extracted_bits.to(device=device, dtype=torch.float32).mul_(2).sub_(1)
    true_sign = true_bits.to(device=device, dtype=torch.float32).mul_(2).sub_(1)

    best_scores = torch.full((num_samples,), -1e9, device=device)
    best_indices = torch.full((num_samples,), -1, device=device, dtype=torch.long)
    true_scores = (extracted_sign * true_sign).sum(dim=1)

    for start in range(0, pool_size, chunk_size):
        end = min(start + chunk_size, pool_size)
        chunk = candidate_pool[start:end].to(device=device, dtype=torch.float32)
        chunk = chunk.mul_(2).sub_(1)
        scores = extracted_sign @ chunk.t()
        chunk_best_scores, chunk_best_offsets = scores.max(dim=1)
        better = chunk_best_scores > best_scores
        best_scores = torch.where(better, chunk_best_scores, best_scores)
        best_indices = torch.where(
            better,
            chunk_best_offsets.to(torch.long) + start,
            best_indices,
        )

    top1_match = ((best_scores + bit_dim) * 0.5) / bit_dim
    true_match = ((true_scores + bit_dim) * 0.5) / bit_dim
    trace_acc = (best_indices.cpu() == torch.arange(num_samples)).float().mean().item()
    return TraceabilityResult(
        trace_accuracy=trace_acc,
        mean_top1_match=top1_match.mean().item(),
        mean_true_match=true_match.mean().item(),
        pool_size=pool_size,
    )

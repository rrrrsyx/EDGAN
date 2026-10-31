from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    step: int = 0,
    best_metric: float | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "step": step,
        "best_metric": best_metric,
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def _adapt_encoder_final_upsample_state(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], bool]:
    state = dict(state_dict)
    target = module.state_dict()
    up_weight_key = "net.up64.0.weight"
    up_bias_key = "net.up64.0.bias"
    proj_weight_key = "net.up64.1.weight"
    proj_bias_key = "net.up64.1.bias"

    old_weight = state.get(up_weight_key)
    new_weight = target.get(up_weight_key)
    projection_weight = target.get(proj_weight_key)
    if (
        old_weight is None
        or new_weight is None
        or projection_weight is None
        or old_weight.ndim != 4
        or new_weight.ndim != 4
        or projection_weight.ndim != 4
        or old_weight.shape == new_weight.shape
        or old_weight.shape[0] != new_weight.shape[0]
        or old_weight.shape[2:] != new_weight.shape[2:]
        or old_weight.shape[1] >= new_weight.shape[1]
    ):
        return state, False

    merged_weight = new_weight.clone()
    old_out_channels = old_weight.shape[1]
    merged_weight[:, :old_out_channels] = old_weight.to(device=merged_weight.device, dtype=merged_weight.dtype)
    state[up_weight_key] = merged_weight

    old_bias = state.get(up_bias_key)
    new_bias = target.get(up_bias_key)
    if old_bias is not None and new_bias is not None and old_bias.ndim == 1 and old_bias.shape[0] < new_bias.shape[0]:
        merged_bias = new_bias.clone()
        merged_bias[: old_bias.shape[0]] = old_bias.to(device=merged_bias.device, dtype=merged_bias.dtype)
        state[up_bias_key] = merged_bias

    identity_projection = torch.zeros_like(projection_weight)
    center_y = identity_projection.shape[-2] // 2
    center_x = identity_projection.shape[-1] // 2
    identity_channels = min(old_out_channels, identity_projection.shape[0], identity_projection.shape[1])
    for channel in range(identity_channels):
        identity_projection[channel, channel, center_y, center_x] = 1
    state[proj_weight_key] = identity_projection
    if proj_bias_key in target:
        state[proj_bias_key] = torch.zeros_like(target[proj_bias_key])

    return state, True


def load_state_dict_compatible(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    adapt_encoder_final_upsample: bool = False,
) -> dict[str, Any]:
    adapted = False
    compatible_state = dict(state_dict)
    if adapt_encoder_final_upsample:
        compatible_state, adapted = _adapt_encoder_final_upsample_state(module, compatible_state)

    target_state = module.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[dict[str, str]] = []
    for key, value in compatible_state.items():
        if key not in target_state:
            skipped_keys.append({"key": key, "reason": "unexpected"})
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_keys.append(
                {
                    "key": key,
                    "reason": f"shape {tuple(value.shape)} != {tuple(target_state[key].shape)}",
                }
            )
            continue
        filtered_state[key] = value

    result = module.load_state_dict(filtered_state, strict=False)
    return {
        "adapted_encoder_final_upsample": adapted,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "skipped_keys": skipped_keys,
    }


def load_codec_state(
    path: str | Path,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    if strict:
        encoder.load_state_dict(checkpoint["encoder"], strict=True)
    else:
        load_state_dict_compatible(
            encoder,
            checkpoint["encoder"],
            adapt_encoder_final_upsample=True,
        )
    decoder.load_state_dict(checkpoint["decoder"], strict=strict)
    return checkpoint

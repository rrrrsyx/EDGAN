from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from tqdm import tqdm

from attacks.noise_attacks import DifferentiableNoiseAttackLayer
from losses.watermark_losses import WatermarkLoss
from models.codec import WatermarkDecoder, WatermarkDiscriminator, WatermarkEncoder
from utils.checkpoint import load_checkpoint, load_state_dict_compatible, save_checkpoint
from utils.randomness_metrics import gaussian_randomness_metrics
from utils.seed import resolve_device, set_seed


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        from torch.cuda.amp import GradScaler

        return GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        from torch.cuda.amp import autocast

        return autocast(enabled=enabled)


class CsvLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] | None = None
        self.rows: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        else:
            for key in row.keys():
                if key not in self.fieldnames:
                    self.fieldnames.append(key)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for item in self.rows:
                writer.writerow({key: item.get(key) for key in self.fieldnames})


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_models(config: dict[str, Any], device: torch.device) -> tuple[WatermarkEncoder, WatermarkDecoder]:
    model_cfg = config.get("model", {})
    encoder = WatermarkEncoder(
        bit_dim=int(model_cfg.get("bit_dim", 128)),
        latent_channels=int(model_cfg.get("latent_channels", 4)),
        latent_size=int(model_cfg.get("latent_size", 64)),
    ).to(device)
    decoder = WatermarkDecoder(
        bit_dim=int(model_cfg.get("bit_dim", 128)),
        latent_channels=int(model_cfg.get("latent_channels", 4)),
        latent_size=int(model_cfg.get("latent_size", 64)),
        base_channels=int(model_cfg.get("decoder_base_channels", 32)),
        decoder_channels=model_cfg.get("decoder_channels"),
    ).to(device)
    return encoder, decoder


def build_discriminator(config: dict[str, Any], device: torch.device) -> WatermarkDiscriminator:
    model_cfg = config.get("model", {})
    discriminator = WatermarkDiscriminator(
        latent_channels=int(model_cfg.get("latent_channels", 4)),
        latent_size=int(model_cfg.get("latent_size", 64)),
        base_channels=int(
            model_cfg.get(
                "discriminator_base_channels",
                model_cfg.get("decoder_base_channels", 32),
            )
        ),
        discriminator_channels=model_cfg.get("discriminator_channels", model_cfg.get("decoder_channels")),
    ).to(device)
    return discriminator


def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def as_logit_tuple(logits: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if isinstance(logits, torch.Tensor):
        return (logits,)
    return tuple(logits)


def is_wgan_loss(loss_type: str) -> bool:
    return loss_type.lower() in {"wgan", "wgan_gp", "wgan-gp"}


def uses_gradient_penalty(loss_type: str) -> bool:
    return loss_type.lower() in {"wgan_gp", "wgan-gp"}


def critic_per_sample(logits: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    values = []
    for item in as_logit_tuple(logits):
        values.append(item.reshape(item.shape[0], -1).mean(dim=1))
    return torch.stack(values, dim=0).mean(dim=0)


def critic_mean(logits: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    return critic_per_sample(logits).mean()


def adversarial_bce_loss(
    logits: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    target: float,
) -> torch.Tensor:
    losses = [
        F.binary_cross_entropy_with_logits(item, torch.full_like(item, float(target)))
        for item in as_logit_tuple(logits)
    ]
    return torch.stack(losses).mean()


def discriminator_adversarial_loss(
    logits: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    real: bool,
    loss_type: str = "hinge",
    margin: float = 1.0,
    target: float | None = None,
) -> torch.Tensor:
    normalized_loss_type = loss_type.lower()
    if normalized_loss_type == "bce":
        if target is None:
            target = 1.0 if real else 0.0
        return adversarial_bce_loss(logits, target)
    if is_wgan_loss(normalized_loss_type):
        return -critic_mean(logits) if real else critic_mean(logits)
    if normalized_loss_type != "hinge":
        raise ValueError(f"Unsupported adversarial loss type: {loss_type}")
    losses = []
    for item in as_logit_tuple(logits):
        if real:
            losses.append(F.relu(float(margin) - item).mean())
        else:
            losses.append(F.relu(float(margin) + item).mean())
    return torch.stack(losses).mean()


def generator_adversarial_loss(
    logits: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    loss_type: str = "hinge",
    target: float = 1.0,
) -> torch.Tensor:
    normalized_loss_type = loss_type.lower()
    if normalized_loss_type == "bce":
        return adversarial_bce_loss(logits, target)
    if is_wgan_loss(normalized_loss_type):
        return -critic_mean(logits)
    if normalized_loss_type != "hinge":
        raise ValueError(f"Unsupported adversarial loss type: {loss_type}")
    losses = [-item.mean() for item in as_logit_tuple(logits)]
    return torch.stack(losses).mean()


def wgan_gradient_penalty(
    discriminator: WatermarkDiscriminator,
    real_noise: torch.Tensor,
    fake_noise: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(real_noise.shape[0])
    alpha_shape = (batch_size,) + (1,) * (real_noise.ndim - 1)
    alpha = torch.rand(alpha_shape, device=real_noise.device, dtype=real_noise.dtype)
    interpolated = (alpha * real_noise.detach() + (1.0 - alpha) * fake_noise.detach()).requires_grad_(True)
    with autocast_context(enabled=False):
        logits = discriminator(interpolated.float())
    scores = critic_per_sample(logits)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return (gradients.norm(2, dim=1) - 1.0).pow(2).mean()


def discriminator_score(
    logits: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    loss_type: str,
) -> torch.Tensor:
    if is_wgan_loss(loss_type):
        return critic_mean(logits).detach()
    scores = [torch.sigmoid(item.detach()).mean() for item in as_logit_tuple(logits)]
    return torch.stack(scores).mean()


def discriminator_accuracy(
    logits: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    real: bool,
) -> torch.Tensor:
    values = []
    for item in as_logit_tuple(logits):
        prediction = item.detach() > 0 if real else item.detach() < 0
        values.append(prediction.float().mean())
    return torch.stack(values).mean()


def wgan_pairwise_accuracy(
    real_logits: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    fake_logits: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> torch.Tensor:
    return (critic_per_sample(real_logits).detach() > critic_per_sample(fake_logits).detach()).float().mean()


def discriminator_lr_scale(
    acc: float,
    *,
    strong_acc: float = 0.75,
    very_strong_acc: float = 0.90,
    min_scale: float = 0.05,
) -> float:
    if acc <= strong_acc:
        return 1.0
    if acc >= very_strong_acc:
        return float(min_scale)
    progress = (acc - strong_acc) / max(very_strong_acc - strong_acc, 1e-8)
    return float(1.0 + progress * (min_scale - 1.0))


def linear_encoder_bit_gate(acc: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0 if acc >= high else 1.0
    value = (high - acc) / (high - low)
    return float(max(0.0, min(1.0, value)))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def maybe_create_tb_writer(output_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(log_dir=str(output_dir / "tb"))


def save_channel_distribution_plot(noise: torch.Tensor, step: int, step_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    channels = int(noise.shape[0])
    cols = 2 if channels > 1 else 1
    rows = math.ceil(channels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.4), squeeze=False)

    flat = noise.flatten()
    plot_min = min(-4.0, float(torch.quantile(flat, 0.001).item()))
    plot_max = max(4.0, float(torch.quantile(flat, 0.999).item()))
    x = torch.linspace(plot_min, plot_max, 512)
    standard_normal_pdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * x.pow(2))
    x_np = x.numpy()
    pdf_np = standard_normal_pdf.numpy()

    for channel in range(channels):
        ax = axes[channel // cols][channel % cols]
        channel_noise = noise[channel]
        values = channel_noise.flatten().numpy()
        mean = float(channel_noise.mean().item())
        std = float(channel_noise.std(unbiased=False).item())
        ax.hist(values, bins=80, range=(plot_min, plot_max), density=True, alpha=0.72, color="#4c78a8")
        ax.plot(x_np, pdf_np, color="#d62728", linewidth=2.0, label="N(0,1)")
        ax.axvline(0.0, color="#222222", linewidth=0.8, alpha=0.45)
        ax.set_title(f"channel {channel}  mean={mean:.3f}, std={std:.3f}")
        ax.set_xlabel("latent value")
        ax.set_ylabel("density")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right", frameon=False)

    for index in range(channels, rows * cols):
        axes[index // cols][index % cols].axis("off")

    fig.suptitle(f"Latent channel distributions at step {step}", y=0.995)
    fig.tight_layout()
    fig.savefig(step_dir / "channel_distributions.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_noise_visualization(
    encoder: WatermarkEncoder,
    bits: torch.Tensor,
    step: int,
    output_dir: Path,
) -> None:
    was_training = encoder.training
    encoder.eval()
    noise = encoder(bits).detach().float().cpu()[0]
    if was_training:
        encoder.train()

    from PIL import Image

    step_dir = output_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "step": step,
        "mapping": "Stable Diffusion postprocess: clamp(noise / 2 + 0.5, 0, 1) * 255",
        "distribution_plot": "channel_distributions.png",
        "channels": {},
    }
    for channel in range(noise.shape[0]):
        channel_noise = noise[channel]
        image = (channel_noise / 2.0 + 0.5).clamp(0.0, 1.0) * 255
        image = image.round().to(torch.uint8).numpy()
        Image.fromarray(image, mode="L").save(step_dir / f"channel_{channel}.png")
        stats["channels"][f"channel_{channel}"] = {
            "mean": float(channel_noise.mean().item()),
            "std": float(channel_noise.std(unbiased=False).item()),
            "min": float(channel_noise.min().item()),
            "max": float(channel_noise.max().item()),
        }
    save_channel_distribution_plot(noise, step, step_dir)
    (step_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def create_timestamped_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ED-DIW latent watermark codec.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--visualize_interval", type=int, default=None)
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=None,
        help="Override loss.lambda_kl from the config for this training run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = dict(config.get("train", {}))
    for key in (
        "total_steps",
        "batch_size",
        "save_interval",
        "log_interval",
        "visualize_interval",
    ):
        value = getattr(args, key)
        if value is not None:
            train_cfg[key] = value
            config.setdefault("train", {})[key] = value
    if args.lambda_kl is not None:
        config.setdefault("loss", {})["lambda_kl"] = float(args.lambda_kl)
    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    betas = tuple(float(x) for x in train_cfg.get("betas", [0.9, 0.999]))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    gan_cfg = dict(config.get("gan", {}))
    gan_cfg.setdefault("lambda_adv", 0.1)
    gan_cfg.setdefault("real_label", 1.0)
    gan_cfg.setdefault("fake_label", 0.0)
    gan_cfg.setdefault("discriminator_learning_rate", learning_rate)
    gan_cfg.setdefault("discriminator_weight_decay", weight_decay)
    gan_cfg.setdefault("discriminator_betas", list(betas))
    gan_cfg.setdefault("discriminator_grad_clip_norm", grad_clip_norm)
    gan_cfg.setdefault("gradient_penalty_weight", 10.0)
    config["gan"] = gan_cfg
    bit_dim = int(config.get("model", {}).get("bit_dim", 128))
    base_output_dir = Path(args.output_dir or train_cfg.get("output_dir", "outputs/codec"))
    output_dir = create_timestamped_output_dir(base_output_dir)
    config.setdefault("train", {})["output_dir"] = str(output_dir)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = resolve_device(args.device or config.get("device", "auto"))
    set_seed(int(config.get("seed", 1234)))

    encoder, decoder = build_models(config, device)
    discriminator = build_discriminator(config, device)
    attack_layer = DifferentiableNoiseAttackLayer(config.get("attack", {})).to(device)
    loss_cfg = dict(config.get("loss", {}))
    criterion_cfg = dict(loss_cfg)
    criterion_cfg["lambda_bit"] = 0.0
    criterion = WatermarkLoss(criterion_cfg)
    lambda_bit_decoder = float(loss_cfg.get("lambda_bit_decoder", loss_cfg.get("lambda_bit", 1.0)))
    lambda_bit_encoder_max = float(loss_cfg.get("lambda_bit_encoder_max", loss_cfg.get("lambda_bit", 1.0)))
    encoder_bit_gate_low = float(loss_cfg.get("encoder_bit_gate_low", 0.85))
    encoder_bit_gate_high = float(loss_cfg.get("encoder_bit_gate_high", 0.98))

    generator_parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer_g = torch.optim.AdamW(
        generator_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(gan_cfg.get("discriminator_learning_rate", learning_rate)),
        weight_decay=float(gan_cfg.get("discriminator_weight_decay", weight_decay)),
        betas=tuple(float(x) for x in gan_cfg.get("discriminator_betas", betas)),
    )

    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = make_grad_scaler(enabled=amp_enabled)
    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        encoder_load_report = load_state_dict_compatible(
            encoder,
            checkpoint["encoder"],
            adapt_encoder_final_upsample=True,
        )
        decoder.load_state_dict(checkpoint["decoder"])
        generator_optimizer_compatible = not (
            encoder_load_report["adapted_encoder_final_upsample"]
            or encoder_load_report["missing_keys"]
            or encoder_load_report["skipped_keys"]
        )
        if encoder_load_report["adapted_encoder_final_upsample"]:
            print(
                "Warning: adapted old encoder final upsample weights to the current "
                "64->32->4 generator output block."
            )
        if encoder_load_report["missing_keys"] or encoder_load_report["skipped_keys"]:
            print("Warning: encoder checkpoint was loaded with missing or skipped keys.")
        extra = checkpoint.get("extra", {})
        discriminator_state = checkpoint.get("discriminator", extra.get("discriminator"))
        discriminator_loaded = False
        if discriminator_state is not None:
            try:
                discriminator.load_state_dict(discriminator_state)
                discriminator_loaded = True
            except RuntimeError as exc:
                print(
                    "Warning: checkpoint discriminator is incompatible with the current "
                    "scalar discriminator; initializing discriminator from scratch."
                )
                print(f"Warning detail: {exc.__class__.__name__}")
        if generator_optimizer_compatible and "optimizer" in checkpoint:
            optimizer_g.load_state_dict(checkpoint["optimizer"])
        elif generator_optimizer_compatible and "optimizer_g" in extra:
            optimizer_g.load_state_dict(extra["optimizer_g"])
        elif not generator_optimizer_compatible:
            print("Warning: generator optimizer state was reset because the generator structure changed.")
        if "optimizer_d" in extra and discriminator_loaded:
            optimizer_d.load_state_dict(extra["optimizer_d"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint.get("step", 0))

    csv_logger = CsvLogger(output_dir / "train_log.csv")
    tb_writer = maybe_create_tb_writer(output_dir)

    total_steps = int(train_cfg.get("total_steps", 3000))
    batch_size = int(train_cfg.get("batch_size", 16))
    log_interval = int(train_cfg.get("log_interval", 20))
    save_interval = int(train_cfg.get("save_interval", 500))
    visualize_interval = int(train_cfg.get("visualize_interval", 200))
    discriminator_grad_clip_norm = float(gan_cfg.get("discriminator_grad_clip_norm", grad_clip_norm))
    discriminator_base_lr = float(gan_cfg.get("discriminator_learning_rate", learning_rate))
    adaptive_d_lr_enabled = bool(gan_cfg.get("adaptive_discriminator_lr", True))
    adaptive_d_lr_strong_acc = float(gan_cfg.get("adaptive_discriminator_strong_acc", 0.75))
    adaptive_d_lr_very_strong_acc = float(gan_cfg.get("adaptive_discriminator_very_strong_acc", 0.90))
    adaptive_d_lr_min_scale = float(gan_cfg.get("adaptive_discriminator_min_lr_scale", 0.05))
    current_discriminator_lr_scale = 1.0
    adversarial_loss_type = str(gan_cfg.get("adversarial_loss", "hinge")).lower()
    hinge_margin = float(gan_cfg.get("hinge_margin", 1.0))
    gradient_penalty_weight = float(gan_cfg.get("gradient_penalty_weight", 10.0))
    lambda_adv = float(gan_cfg.get("lambda_adv", 0.1))
    real_label = float(gan_cfg.get("real_label", 1.0))
    fake_label = float(gan_cfg.get("fake_label", 0.0))
    randomness_eval_cfg = config.get("randomness_eval", {})
    randomness_eval_enabled = bool(randomness_eval_cfg.get("enabled", True))
    randomness_eval_max_samples = int(randomness_eval_cfg.get("max_samples", 128))
    randomness_eval_eps = float(randomness_eval_cfg.get("eps", 1e-6))
    vis_generator = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 1234)) + 4040)
    vis_bits = torch.randint(0, 2, (1, bit_dim), generator=vis_generator).to(device=device, dtype=torch.float32)
    noise_vis_dir = output_dir / "noise_vis"

    encoder.train()
    decoder.train()
    discriminator.train()
    progress = tqdm(range(start_step + 1, total_steps + 1), desc="train", dynamic_ncols=True)
    for step in progress:
        if adaptive_d_lr_enabled:
            set_optimizer_lr(optimizer_d, discriminator_base_lr * current_discriminator_lr_scale)
        bits = torch.randint(0, 2, (batch_size, bit_dim), device=device).float()

        set_requires_grad(discriminator, True)
        optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_noise_for_d = encoder(bits)
        real_noise = torch.randn_like(fake_noise_for_d)
        with autocast_context(enabled=amp_enabled):
            real_logits = discriminator(real_noise)
            fake_logits = discriminator(fake_noise_for_d.detach())
            d_loss_real = discriminator_adversarial_loss(
                real_logits,
                real=True,
                loss_type=adversarial_loss_type,
                margin=hinge_margin,
                target=real_label,
            )
            d_loss_fake = discriminator_adversarial_loss(
                fake_logits,
                real=False,
                loss_type=adversarial_loss_type,
                margin=hinge_margin,
                target=fake_label,
            )
            if is_wgan_loss(adversarial_loss_type):
                gradient_penalty = wgan_gradient_penalty(
                    discriminator,
                    real_noise.float(),
                    fake_noise_for_d.detach().float(),
                ) if uses_gradient_penalty(adversarial_loss_type) else real_noise.new_zeros(())
                d_loss = d_loss_real + d_loss_fake + gradient_penalty_weight * gradient_penalty
            else:
                gradient_penalty = real_noise.new_zeros(())
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

        scaler.scale(d_loss).backward()
        if discriminator_grad_clip_norm > 0:
            scaler.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), discriminator_grad_clip_norm)
        scaler.step(optimizer_d)

        set_requires_grad(discriminator, False)
        set_requires_grad(decoder, True)
        optimizer_g.zero_grad(set_to_none=True)
        with autocast_context(enabled=amp_enabled):
            noise = encoder(bits)
            # Training uses latent/noise-domain attacks as an efficient proxy for
            # expensive image generation + DDIM inversion corruptions.
            decoder_noise_attacked = attack_layer(noise.detach())
            decoder_logits = decoder(decoder_noise_attacked)
            loss_bit_decoder = F.binary_cross_entropy_with_logits(decoder_logits, bits)
            codec_loss, metrics = criterion(decoder_logits.detach(), bits, noise)
            encoder_bit_gate_value = linear_encoder_bit_gate(
                float(metrics["watermark_bit_acc"].detach().item()),
                low=encoder_bit_gate_low,
                high=encoder_bit_gate_high,
            )
            lambda_bit_encoder_effective = lambda_bit_encoder_max * encoder_bit_gate_value
            if lambda_bit_encoder_effective > 0.0:
                set_requires_grad(decoder, False)
                encoder_noise_attacked = attack_layer(noise)
                encoder_logits = decoder(encoder_noise_attacked)
                loss_bit_encoder = F.binary_cross_entropy_with_logits(encoder_logits, bits)
                set_requires_grad(decoder, True)
            else:
                loss_bit_encoder = noise.new_zeros(())
            generator_logits = discriminator(noise)
            loss_adv = generator_adversarial_loss(
                generator_logits,
                loss_type=adversarial_loss_type,
                target=real_label,
            )
            loss_bit_decoder_weighted = lambda_bit_decoder * loss_bit_decoder
            loss_bit_encoder_weighted = lambda_bit_encoder_effective * loss_bit_encoder
            loss = codec_loss + loss_bit_decoder_weighted + loss_bit_encoder_weighted + lambda_adv * loss_adv

        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(generator_parameters, grad_clip_norm)
        scaler.step(optimizer_g)
        scaler.update()
        set_requires_grad(discriminator, True)
        set_requires_grad(decoder, True)

        metrics["loss_bit"] = loss_bit_decoder.detach()
        metrics["loss_bit_decoder"] = loss_bit_decoder.detach()
        metrics["loss_bit_encoder"] = loss_bit_encoder.detach()
        metrics["loss_bit_decoder_weighted"] = loss_bit_decoder_weighted.detach()
        metrics["loss_bit_encoder_weighted"] = loss_bit_encoder_weighted.detach()
        metrics["encoder_bit_gate"] = torch.tensor(
            encoder_bit_gate_value,
            device=device,
            dtype=torch.float32,
        )
        metrics["lambda_bit_decoder_effective"] = torch.tensor(
            lambda_bit_decoder,
            device=device,
            dtype=torch.float32,
        )
        metrics["lambda_bit_encoder_effective"] = torch.tensor(
            lambda_bit_encoder_effective,
            device=device,
            dtype=torch.float32,
        )
        metrics["loss_codec"] = codec_loss.detach()
        metrics["loss_adv"] = loss_adv.detach()
        metrics["loss_total"] = loss.detach()
        metrics["loss_discriminator"] = d_loss.detach()
        metrics["loss_discriminator_real"] = d_loss_real.detach()
        metrics["loss_discriminator_fake"] = d_loss_fake.detach()
        metrics["loss_gradient_penalty"] = gradient_penalty.detach()
        metrics["discriminator_real_score"] = discriminator_score(real_logits, loss_type=adversarial_loss_type)
        metrics["discriminator_fake_score"] = discriminator_score(fake_logits, loss_type=adversarial_loss_type)
        metrics["discriminator_generator_score"] = discriminator_score(
            generator_logits,
            loss_type=adversarial_loss_type,
        )
        if is_wgan_loss(adversarial_loss_type):
            metrics["discriminator_real_acc"] = wgan_pairwise_accuracy(real_logits, fake_logits)
            metrics["discriminator_fake_acc"] = metrics["discriminator_real_acc"]
            metrics["discriminator_acc"] = metrics["discriminator_real_acc"]
        else:
            metrics["discriminator_real_acc"] = discriminator_accuracy(real_logits, real=True)
            metrics["discriminator_fake_acc"] = discriminator_accuracy(fake_logits, real=False)
            metrics["discriminator_acc"] = 0.5 * (
                metrics["discriminator_real_acc"] + metrics["discriminator_fake_acc"]
            )
        metrics["discriminator_lr_scale"] = torch.tensor(
            current_discriminator_lr_scale,
            device=device,
            dtype=torch.float32,
        )
        if adaptive_d_lr_enabled:
            current_discriminator_lr_scale = discriminator_lr_scale(
                float(metrics["discriminator_acc"].detach().item()),
                strong_acc=adaptive_d_lr_strong_acc,
                very_strong_acc=adaptive_d_lr_very_strong_acc,
                min_scale=adaptive_d_lr_min_scale,
            )

        if step % log_interval == 0 or step == 1:
            if randomness_eval_enabled:
                with torch.no_grad():
                    metrics.update(
                        gaussian_randomness_metrics(
                            noise,
                            max_samples=randomness_eval_max_samples,
                            eps=randomness_eval_eps,
                        )
                    )
            row = {
                "step": step,
                "lr": optimizer_g.param_groups[0]["lr"],
                "lr_discriminator": optimizer_d.param_groups[0]["lr"],
            }
            row.update({key: float(value.item()) for key, value in metrics.items()})
            csv_logger.write(row)
            if tb_writer is not None:
                for key, value in row.items():
                    if key != "step":
                        tb_writer.add_scalar(key, value, step)
            progress.set_postfix(
                total=row["loss_total"],
                codec=row["loss_codec"],
                adv=row["loss_adv"],
                d=row["loss_discriminator"],
                bit=row["loss_bit"],
                dist=row["loss_dist"],
                kl=row["loss_kl"],
                tail=row["loss_tail"],
                fent=row["freq_entropy"],
                band=row["loss_spectral_band"],
                sspec=row["loss_sign_spectrum"],
                mi=row["loss_mi"],
                spcc=row["same_position_channel_corr_abs_mean"],
                mean=row["loss_mean"],
                bpkl=row["loss_batch_position_kl"],
                bvar=row["batch_position_var_std"],
                fac=row["loss_fft_autocorr"],
                facpk=row["fft_autocorr_peak_loss"],
                facmax=row["fft_autocorr_abs_max"],
                acc=row["watermark_bit_acc"],
                ebw=row["lambda_bit_encoder_effective"],
                dacc=row["discriminator_acc"],
                rsc=row.get("rand_spatial_corr1_abs_mean", 0.0),
                rch=row.get("rand_channel_corr_abs_mean", 0.0),
                rpk=row.get("rand_fft_peak_ratio", 0.0),
            )

        if visualize_interval > 0 and (step % visualize_interval == 0 or step == 1 or step == total_steps):
            save_noise_visualization(
                encoder=encoder,
                bits=vis_bits,
                step=step,
                output_dir=noise_vis_dir,
            )

        if step % save_interval == 0 or step == total_steps:
            checkpoint_extra = {
                "training_mode": "gan_scalar",
                "discriminator_type": "scalar",
                "adversarial_loss": adversarial_loss_type,
                "discriminator": discriminator.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
            }
            save_checkpoint(
                output_dir / "latest.pt",
                encoder,
                decoder,
                optimizer=optimizer_g,
                scaler=scaler,
                step=step,
                config=config,
                extra=checkpoint_extra,
            )
            save_checkpoint(
                output_dir / "checkpoints" / f"step_{step:07d}.pt",
                encoder,
                decoder,
                optimizer=optimizer_g,
                scaler=scaler,
                step=step,
                config=config,
                extra=checkpoint_extra,
            )

    if tb_writer is not None:
        tb_writer.close()
    print("Training finished.")
    print(f"Latest checkpoint: {output_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()

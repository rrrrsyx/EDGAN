from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ResidualRefine(nn.Module):
    def __init__(self, channels: int, scale: float = 1.0, zero_init: bool = False) -> None:
        super().__init__()
        self.scale = float(scale)
        self.net = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        if zero_init:
            nn.init.zeros_(self.net.weight)
            if self.net.bias is not None:
                nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.net(x)


class CenterCrop2d(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = int(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height < self.size or width < self.size:
            raise ValueError(f"input spatial size {(height, width)} is smaller than crop size {self.size}")
        top = (height - self.size) // 2
        left = (width - self.size) // 2
        return x[:, :, top : top + self.size, left : left + self.size]


def upsample_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(1, out_channels),
        nn.SiLU(inplace=True),
    )


class MultiScaleLatentHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        latent_size: int = 64,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        self.proj = nn.Conv2d(
            in_channels,
            latent_channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if x.shape[-2:] != (self.latent_size, self.latent_size):
            x = F.interpolate(x, size=(self.latent_size, self.latent_size), mode="bilinear", align_corners=False)
        return x


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def resolve_decoder_channels(
    base_channels: int = 32,
    decoder_channels: Sequence[int] | None = None,
) -> tuple[int, int, int, int]:
    if decoder_channels is None:
        channels = (
            int(base_channels),
            int(base_channels) * 2,
            int(base_channels) * 4,
            int(base_channels) * 8,
        )
    else:
        channels = tuple(int(value) for value in decoder_channels)
    if len(channels) != 4:
        raise ValueError("decoder_channels must contain exactly 4 increasing channel sizes")
    if any(value <= 0 for value in channels):
        raise ValueError("decoder channel sizes must be positive")
    if not (channels[0] < channels[1] < channels[2] < channels[3]):
        raise ValueError("decoder channel sizes must be strictly increasing")
    return channels


class MultiScaleEncoderNet(nn.Module):
    """Encoder with multi-scale latent residual predictions."""

    def __init__(self, bit_dim: int = 128, latent_channels: int = 4, latent_size: int = 64) -> None:
        super().__init__()
        if latent_size != 64:
            raise ValueError("MultiScaleEncoderNet expects latent_size=64")
        self.fc = nn.Sequential(
            # Bit vector -> small 4x4 feature map.
            nn.Linear(bit_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(inplace=True),
            nn.Linear(1024, 4096),
            nn.LayerNorm(4096),
            nn.SiLU(inplace=True),
            nn.Linear(4096, 512 * 4 * 4),
            nn.SiLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),
        )
        # Main path: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 68x68 -> center crop 64x64.
        self.up8 = upsample_conv_block(512, 256)
        self.refine8 = ResidualRefine(256, zero_init=True)
        self.up16 = upsample_conv_block(256, 128)
        self.refine16 = ResidualRefine(128, zero_init=True)
        self.up32 = upsample_conv_block(128, 64)
        self.refine32 = ResidualRefine(64, zero_init=True)
        self.up64 = nn.Sequential(
            nn.ConvTranspose2d(64, latent_channels, kernel_size=8, stride=2, padding=1),
            CenterCrop2d(latent_size),
        )

        # Each scale predicts a latent-sized residual. These are zero-init so the
        # initial behavior matches the old single-output encoder.
        self.head4 = MultiScaleLatentHead(512, latent_channels, latent_size=latent_size, zero_init=True)
        self.head8 = MultiScaleLatentHead(256, latent_channels, latent_size=latent_size, zero_init=True)
        self.head16 = MultiScaleLatentHead(128, latent_channels, latent_size=latent_size, zero_init=True)
        self.head32 = MultiScaleLatentHead(64, latent_channels, latent_size=latent_size, zero_init=True)
        self.head64 = MultiScaleLatentHead(latent_channels, latent_channels, latent_size=latent_size, zero_init=True)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        x4 = self.fc(bits)
        x8 = self.refine8(self.up8(x4))
        x16 = self.refine16(self.up16(x8))
        x32 = self.refine32(self.up32(x16))
        x64 = self.up64(x32)
        noise = (
            x64
            + self.head4(x4)
            + self.head8(x8)
            + self.head16(x16)
            + self.head32(x32)
            + self.head64(x64)
        )
        return noise


def build_encoder(bit_dim: int = 128, latent_channels: int = 4, latent_size: int = 64) -> nn.Module:
    """Encoder structure: [B, bit_dim] -> [B, latent_channels, 64, 64]."""
    return MultiScaleEncoderNet(bit_dim=bit_dim, latent_channels=latent_channels, latent_size=latent_size)


def build_decoder(
    bit_dim: int = 128,
    latent_channels: int = 4,
    base_channels: int = 32,
    decoder_channels: Sequence[int] | None = None,
) -> nn.Sequential:
    """Decoder structure: [B, latent_channels, 64, 64] -> [B, bit_dim]."""
    c1, c2, c3, c4 = resolve_decoder_channels(base_channels, decoder_channels)
    hidden = c4 * 4
    return nn.Sequential(
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4.
        nn.Conv2d(latent_channels, c1, kernel_size=4, stride=2, padding=1),
        group_norm(c1),
        nn.SiLU(inplace=True),
        nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
        group_norm(c2),
        nn.SiLU(inplace=True),
        nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
        group_norm(c3),
        nn.SiLU(inplace=True),
        nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1),
        group_norm(c4),
        nn.SiLU(inplace=True),
        # 4x4 feature map -> 128 independent bit logits.
        nn.Flatten(),
        nn.Linear(c4 * 4 * 4, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(inplace=True),
        nn.Linear(hidden, bit_dim),
    )


def discriminator_down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="circular",
        ),
        group_norm(out_channels),
        nn.SiLU(inplace=True),
    )


class DiscriminatorResidualBlock(nn.Module):
    """Single-conv residual refinement at a fixed discriminator scale."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
            group_norm(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ScalarDiscriminatorNet(nn.Module):
    """Single-output discriminator for latent noise."""

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 32,
        discriminator_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = resolve_decoder_channels(base_channels, discriminator_channels)
        self.down32 = discriminator_down_block(latent_channels, c1)
        self.res32 = DiscriminatorResidualBlock(c1)
        self.down16 = discriminator_down_block(c1, c2)
        self.res16 = DiscriminatorResidualBlock(c2)
        self.down8 = discriminator_down_block(c2, c3)
        self.res8 = DiscriminatorResidualBlock(c3)
        self.down4 = discriminator_down_block(c3, c4)
        self.res4 = DiscriminatorResidualBlock(c4)
        self.global_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4 * 4 * 4, c4),
            nn.SiLU(inplace=True),
            nn.Linear(c4, 1),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x32 = self.res32(self.down32(noise))
        x16 = self.res16(self.down16(x32))
        x8 = self.res8(self.down8(x16))
        x4 = self.res4(self.down4(x8))
        return self.global_head(x4)


def build_discriminator(
    latent_channels: int = 4,
    base_channels: int = 32,
    discriminator_channels: Sequence[int] | None = None,
) -> nn.Module:
    """Discriminator structure: [B, latent_channels, 64, 64] -> [B, 1]."""
    return ScalarDiscriminatorNet(
        latent_channels=latent_channels,
        base_channels=base_channels,
        discriminator_channels=discriminator_channels,
    )


class WatermarkEncoder(nn.Module):
    """Maps a watermark bit vector to a Stable-Diffusion-sized latent noise tensor."""

    def __init__(
        self,
        bit_dim: int = 128,
        latent_channels: int = 4,
        latent_size: int = 64,
    ) -> None:
        super().__init__()
        if latent_size != 64:
            raise ValueError("This simple architecture expects latent_size=64")
        self.bit_dim = bit_dim
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.net = build_encoder(bit_dim=bit_dim, latent_channels=latent_channels, latent_size=latent_size)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[1] != self.bit_dim:
            raise ValueError(f"bits must have shape [B,{self.bit_dim}]")
        return self.net(bits.float())


class WatermarkDecoder(nn.Module):
    """Maps a latent noise tensor back to independent bit logits."""

    def __init__(
        self,
        bit_dim: int = 128,
        latent_channels: int = 4,
        latent_size: int = 64,
        base_channels: int = 32,
        decoder_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if latent_size != 64:
            raise ValueError("This simple architecture expects latent_size=64")
        self.bit_dim = bit_dim
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.base_channels = base_channels
        self.decoder_channels = resolve_decoder_channels(base_channels, decoder_channels)
        self.net = build_decoder(
            bit_dim=bit_dim,
            latent_channels=latent_channels,
            base_channels=base_channels,
            decoder_channels=self.decoder_channels,
        )

    def forward(self, noise_attacked: torch.Tensor) -> torch.Tensor:
        expected = (self.latent_channels, self.latent_size, self.latent_size)
        if noise_attacked.ndim != 4 or tuple(noise_attacked.shape[1:]) != expected:
            raise ValueError(f"noise_attacked must have shape [B,{expected[0]},{expected[1]},{expected[2]}]")
        return self.net(noise_attacked.float())


class WatermarkDiscriminator(nn.Module):
    """Classifies real Gaussian latent noise versus encoder-generated noise at multiple scales."""

    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 64,
        base_channels: int = 32,
        discriminator_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if latent_size != 64:
            raise ValueError("This simple architecture expects latent_size=64")
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.base_channels = base_channels
        self.discriminator_channels = resolve_decoder_channels(base_channels, discriminator_channels)
        self.net = build_discriminator(
            latent_channels=latent_channels,
            base_channels=base_channels,
            discriminator_channels=self.discriminator_channels,
        )

    def forward(self, noise: torch.Tensor) -> tuple[torch.Tensor, ...]:
        expected = (self.latent_channels, self.latent_size, self.latent_size)
        if noise.ndim != 4 or tuple(noise.shape[1:]) != expected:
            raise ValueError(f"noise must have shape [B,{expected[0]},{expected[1]},{expected[2]}]")
        return self.net(noise.float())


class WatermarkCodec(nn.Module):
    def __init__(
        self,
        bit_dim: int = 128,
        latent_channels: int = 4,
        latent_size: int = 64,
        decoder_base_channels: int = 32,
        decoder_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = WatermarkEncoder(
            bit_dim=bit_dim,
            latent_channels=latent_channels,
            latent_size=latent_size,
        )
        self.decoder = WatermarkDecoder(
            bit_dim=bit_dim,
            latent_channels=latent_channels,
            latent_size=latent_size,
            base_channels=decoder_base_channels,
            decoder_channels=decoder_channels,
        )

    def forward(self, bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = self.encoder(bits)
        logits = self.decoder(noise)
        return noise, logits

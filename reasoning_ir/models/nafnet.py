"""Core NAFNet architecture components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps.

    The module normalizes each channel independently across the spatial
    dimensions and learns an affine transform similar to ``nn.LayerNorm`` with a
    ``normalized_shape`` equal to the number of channels.
    """

    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return x_norm * weight + bias


class SimpleGate(nn.Module):
    """Implements the element-wise gating operation used in NAFNet blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Single block of the NAFNet backbone.

    The implementation follows the structure proposed in the paper *Simple
    Baselines for Image Restoration* with depth-wise convolutions, simplified
    gating and residual connections. Compared to the official implementation the
    module is intentionally compact and well documented to make experimentation
    easier in research settings.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.simple_gate = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True),
        )
        self.pwconv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.ffn_gate = SimpleGate()
        self.ffn2 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        shortcut = x
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.dwconv(x)
        x = self.simple_gate(x)
        x = self.sca(x) * x
        x = self.pwconv2(x)
        x = self.dropout1(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ffn1(x)
        x = self.ffn_gate(x)
        x = self.ffn2(x)
        x = self.dropout2(x)
        return shortcut + x


@dataclass
class NAFNetConfig:
    """Configuration values for :class:`NAFNet`."""

    img_channels: int = 3
    width: int = 32
    enc_blk_nums: Sequence[int] = (2, 2, 4, 8)
    middle_blk_num: int = 1
    dec_blk_nums: Sequence[int] = (2, 2, 2, 2)
    dw_expand: int = 2
    ffn_expand: int = 2
    drop_out_rate: float = 0.0

    def __post_init__(self) -> None:
        if len(self.enc_blk_nums) != len(self.dec_blk_nums):
            raise ValueError("Encoder and decoder block counts must be the same length.")


class NAFNet(nn.Module):
    """Implementation of the NAFNet encoder/decoder backbone."""

    def __init__(self, config: NAFNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NAFNetConfig()
        c = self.config.width

        self.intro = nn.Conv2d(self.config.img_channels, c, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        in_channels = c
        for num_blocks in self.config.enc_blk_nums:
            blocks = [
                NAFBlock(
                    in_channels,
                    dw_expand=self.config.dw_expand,
                    ffn_expand=self.config.ffn_expand,
                    drop_out_rate=self.config.drop_out_rate,
                )
                for _ in range(num_blocks)
            ]
            self.encoders.append(nn.Sequential(*blocks))
            self.downs.append(nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2))
            in_channels *= 2

        middle_blocks = [
            NAFBlock(
                in_channels,
                dw_expand=self.config.dw_expand,
                ffn_expand=self.config.ffn_expand,
                drop_out_rate=self.config.drop_out_rate,
            )
            for _ in range(self.config.middle_blk_num)
        ]
        self.middle = nn.Sequential(*middle_blocks)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num_blocks in self.config.dec_blk_nums:
            self.ups.append(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            )
            in_channels //= 2
            blocks = [
                NAFBlock(
                    in_channels,
                    dw_expand=self.config.dw_expand,
                    ffn_expand=self.config.ffn_expand,
                    drop_out_rate=self.config.drop_out_rate,
                )
                for _ in range(num_blocks)
            ]
            self.decoders.append(nn.Sequential(*blocks))

        self.outro = nn.Conv2d(in_channels, self.config.img_channels, kernel_size=3, padding=1)

    @property
    def latent_channels(self) -> int:
        """Number of channels in the highest resolution latent space."""

        return self.config.width

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Run the image through the network.

        Args:
            x: Input image tensor.
            hidden: Optional latent feature tensor carried from a previous pass.
            return_latent: Whether to also return the highest-resolution latent
                representation before the final convolution.
        """

        residual = x
        x = self.intro(x)
        if hidden is not None:
            x = x + hidden

        enc_feats: List[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            enc_feats.append(x)
            x = down(x)

        x = self.middle(x)

        for up, decoder in zip(self.ups, self.decoders):
            x = up(x)
            skip = enc_feats.pop()
            x = x + skip
            x = decoder(x)

        latent = x
        x = self.outro(x)
        out = x + residual
        if return_latent:
            return out, latent
        return out

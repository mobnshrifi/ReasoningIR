"""Recurrent wrapper adding reasoning iterations on top of NAFNet."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from .nafnet import NAFNet, NAFNetConfig


@dataclass
class RecurrentConfig:
    """Configuration for :class:`RecurrentNAFNet`."""

    num_iterations: int = 4
    blend_degraded: bool = True


class RecurrentNAFNet(nn.Module):
    """Wraps :class:`NAFNet` with latent-space recurrence.

    The wrapper introduces a lightweight reasoning loop that iteratively refines
    the restoration result. The refinement happens in the latent space produced
    by the NAFNet intro layer which keeps the computational overhead minimal.
    The design is inspired by the feedback mechanism from RFR-Net but leverages
    the simplicity of NAFNet blocks.
    """

    def __init__(
        self,
        base_config: NAFNetConfig | None = None,
        recurrent_config: RecurrentConfig | None = None,
    ) -> None:
        super().__init__()
        self.base = NAFNet(base_config)
        self.recurrent_config = recurrent_config or RecurrentConfig()
        in_channels = self.base.config.img_channels
        self.input_merger = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.hidden_merger = nn.Conv2d(
            self.base.latent_channels * 2,
            self.base.latent_channels,
            kernel_size=1,
        )

    @property
    def num_iterations(self) -> int:
        return self.recurrent_config.num_iterations

    def forward(
        self,
        degraded: torch.Tensor,
        num_iterations: int | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:  # type: ignore[override]
        """Restore ``degraded`` images using iterative latent reasoning.

        Args:
            degraded: Batch of degraded images.
            num_iterations: Optional override for the number of recurrent
                refinement steps.
            return_sequence: When ``True`` all intermediate predictions are
                returned as well.
        """

        steps = num_iterations or self.recurrent_config.num_iterations
        outputs: List[torch.Tensor] = []
        prev_output: torch.Tensor | None = None
        hidden: torch.Tensor | None = None

        for _ in range(steps):
            if prev_output is None or not self.recurrent_config.blend_degraded:
                model_input = degraded
            else:
                merged = torch.cat([degraded, prev_output], dim=1)
                model_input = self.input_merger(merged)

            restored, latent = self.base(model_input, hidden=hidden, return_latent=True)

            if hidden is None:
                hidden = latent
            else:
                hidden = self.hidden_merger(torch.cat([hidden, latent], dim=1))

            outputs.append(restored)
            prev_output = restored

        final = outputs[-1]
        if return_sequence:
            return final, outputs
        return final

    def freeze_base(self) -> None:
        """Convenience helper to freeze the base NAFNet parameters."""

        for param in self.base.parameters():
            param.requires_grad_(False)

    def unfreeze_base(self) -> None:
        """Undo :meth:`freeze_base`."""

        for param in self.base.parameters():
            param.requires_grad_(True)

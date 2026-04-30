#!/usr/bin/env python3
"""
Delta-Encoder model definitions.

Architecture:
  Encoder: Input(120) -> Dense(4096, LeakyReLU alpha=0.3) -> Dense(5) -> delta-vector
  Decoder: Concat(delta-vector[5], sampleRef[120]) -> Dense(4096, LeakyReLU alpha=0.3) -> Dense(120)
"""

import torch
import torch.nn as nn


class DeltaEncoder(nn.Module):
    """
    Encoder: map sampleInput(120) to delta-vector(5)
    to capture intra-class variations.
    """
    def __init__(self, emb_dim: int = 120, delta_dim: int = 5,
                 hidden: int = 4096, alpha: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Linear(hidden, delta_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 120) -> delta-vector: (B, 5)."""
        return self.net(x)


class DeltaDecoder(nn.Module):
    """
    Decoder: reconstruct sampleInput(120) from delta-vector(5) + sampleRef(120).
    """
    def __init__(self, emb_dim: int = 120, delta_dim: int = 5,
                 hidden: int = 4096, alpha: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + delta_dim, hidden),   # 125 → 4096
            nn.LeakyReLU(alpha, inplace=True),
            nn.Linear(hidden, emb_dim),               # 4096 → 120
        )

    def forward(self, delta: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        delta : (B, 5)
        ref   : (B, 120)
        output: (B, 120)
        """
        x = torch.cat([delta, ref], dim=-1)   # (B, 125)
        return self.net(x)


class DeltaEncoderModel(nn.Module):
    """
    Full Delta-Encoder (Encoder + Decoder).

    Training:
      sampleInput -> Encoder -> delta-vector
      Concat(delta-vector, sampleRef) -> Decoder -> reconstruction(120)
      Loss = MSE(reconstruction, sampleInput)

    Inference (sample synthesis):
      any delta-vector + new sampleRef -> Decoder -> synthesized sample(120)
    """
    def __init__(self, emb_dim: int = 120, delta_dim: int = 5,
                 hidden: int = 4096, alpha: float = 0.3):
        super().__init__()
        self.encoder = DeltaEncoder(emb_dim, delta_dim, hidden, alpha)
        self.decoder = DeltaDecoder(emb_dim, delta_dim, hidden, alpha)

    def forward(self, sample_input: torch.Tensor,
                sample_ref: torch.Tensor) -> torch.Tensor:
        """
        sample_input : (B, 120)
        sample_ref   : (B, 120)
        output       : (B, 120)
        """
        delta = self.encoder(sample_input)
        return self.decoder(delta, sample_ref)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract delta-vector. x: (B, 120) -> (B, 5)."""
        return self.encoder(x)

    def decode(self, delta: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Synthesize samples. delta: (B,5), ref: (B,120) -> (B,120)."""
        return self.decoder(delta, ref)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

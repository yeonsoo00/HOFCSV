"""Multi-scale embedding for chromosome-relative SV coordinates."""

import math

import torch
import torch.nn as nn


class GenomicCoordinateEmbedding(nn.Module):
    """Fourier embedding of normalized SV start, end, center, and span."""

    def __init__(self, embedding_dim=64, fourier_bands=12, dropout=0.15):
        super().__init__()
        frequencies = 2.0 ** torch.arange(fourier_bands, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        input_dim = 3 * fourier_bands * 2 + 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, coordinates):
        positions = coordinates[..., :3].clamp(0.0, 1.0)
        angles = 2.0 * math.pi * positions.unsqueeze(-1) * self.frequencies
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        fourier = fourier.flatten(start_dim=-2)
        span = coordinates[..., 3:4].clamp(0.0, 1.0)
        log_span = (torch.log10(span.clamp_min(1e-9)) + 9.0) / 9.0
        return self.projection(torch.cat([fourier, span, log_span], dim=-1))

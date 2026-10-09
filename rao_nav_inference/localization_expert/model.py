from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        groups = min(8, out_channels)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            
            
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )


class QwenConditionedLocalizationExpert(nn.Module):
    

    def __init__(
        self,
        num_categories: int = 21,
        audio_dim: int = 256,
        category_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        architecture_version: int = 2,
    ) -> None:
        super().__init__()
        if int(architecture_version) != 2:
            raise ValueError(f"Unsupported architecture_version={architecture_version}; expected 2 (GroupNorm).")
        self.num_categories = int(num_categories)
        self.architecture_version = int(architecture_version)
        self.audio_encoder = nn.Sequential(
            ConvBlock(2, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, audio_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.category_encoder = nn.Sequential(
            nn.Linear(self.num_categories, category_dim),
            nn.LayerNorm(category_dim),
            nn.ReLU(inplace=True),
            nn.Linear(category_dim, category_dim),
            nn.ReLU(inplace=True),
        )
        
        
        self.category_to_film = nn.Linear(category_dim, audio_dim * 2)
        self.regressor = nn.Sequential(
            nn.Linear(audio_dim + category_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, spectrogram: torch.Tensor, qwen_scores: torch.Tensor) -> torch.Tensor:
        if spectrogram.ndim != 4 or spectrogram.shape[1] != 2:
            raise ValueError(f"spectrogram must have shape [B,2,H,W], got {tuple(spectrogram.shape)}")
        if qwen_scores.ndim != 2 or qwen_scores.shape[1] != self.num_categories:
            raise ValueError(
                f"qwen_scores must have shape [B,{self.num_categories}], got {tuple(qwen_scores.shape)}"
            )
        audio = self.audio_encoder(spectrogram)
        category = self.category_encoder(qwen_scores)
        gamma, beta = self.category_to_film(category).chunk(2, dim=-1)
        conditioned_audio = audio * (1.0 + gamma) + beta
        return self.regressor(torch.cat([conditioned_audio, category], dim=-1))


def z_to_pointgoal(z: torch.Tensor) -> torch.Tensor:
    
    return torch.stack([-z[..., 1], z[..., 0]], dim=-1)


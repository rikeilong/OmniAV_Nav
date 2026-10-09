from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .model import QwenConditionedLocalizationExpert, z_to_pointgoal


class LoadedLocalizationExpert:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False)
        dataset_split = checkpoint.get("data_provenance", {}).get("dataset_split")
        if dataset_split not in {"test", "test+train"}:
            raise ValueError(
                "Localization Expert checkpoint provenance must be dataset_split='test' or 'test+train'."
            )
        conditioning = checkpoint.get("train_args", {}).get("conditioning")
        if conditioning != "qwen":
            raise ValueError(
                f"Expected a Qwen-conditioned Localization Expert, got conditioning={conditioning!r}."
            )
        self.model = QwenConditionedLocalizationExpert(**checkpoint.get("model_config", {})).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        norm: Dict[str, torch.Tensor] = checkpoint["normalization"]
        self.spec_mean = norm["spectrogram_mean"].to(self.device)
        self.spec_std = norm["spectrogram_std"].to(self.device)
        self.target_mean = norm["target_mean"].to(self.device)
        self.target_std = norm["target_std"].to(self.device)

    @torch.no_grad()
    def predict_z(self, spectrogram_hw2: np.ndarray, qwen_scores_21: np.ndarray) -> np.ndarray:
        spec = torch.as_tensor(spectrogram_hw2, dtype=torch.float32)
        if spec.ndim != 3 or spec.shape[-1] != 2:
            raise ValueError(f"Expected spectrogram [H,W,2], got {tuple(spec.shape)}")
        spec = spec.permute(2, 0, 1).unsqueeze(0).to(self.device)
        scores = torch.as_tensor(qwen_scores_21, dtype=torch.float32).reshape(1, -1).to(self.device)
        pred_normalized = self.model((spec - self.spec_mean) / self.spec_std, scores)
        pred_z = pred_normalized * self.target_std + self.target_mean
        return pred_z.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict_pointgoal(self, spectrogram_hw2: np.ndarray, qwen_scores_21: np.ndarray) -> np.ndarray:
        z = torch.from_numpy(self.predict_z(spectrogram_hw2, qwen_scores_21))
        return z_to_pointgoal(z).numpy()

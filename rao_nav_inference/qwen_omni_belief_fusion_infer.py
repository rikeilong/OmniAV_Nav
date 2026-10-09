from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
import torch
import torchvision.models as models
from PIL import Image


TASK_CATEGORIES_21 = [
    "chair",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "sofa",
    "bed",
    "chest_of_drawers",
    "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes",
]


def parse_top3_from_text(text: str, allowed_classes: List[str]) -> List[str]:
    value = (text or "").strip().lower()
    tokens = [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]
    result: List[str] = []
    for token in tokens:
        normalized = re.sub(r"[^a-z_]", "", token)
        if normalized in allowed_classes and normalized not in result:
            result.append(normalized)
        if len(result) == 3:
            break
    if len(result) < 3:
        for category in allowed_classes:
            if re.search(rf"\b{re.escape(category)}\b", value) and category not in result:
                result.append(category)
            if len(result) == 3:
                break
    return result[:3]


def _ensure_placeholder_image(path_hint: str = "") -> str:
    if path_hint:
        path = Path(path_hint)
        if path.exists():
            return path.as_posix()
    import os

    file_descriptor, output_path = tempfile.mkstemp(prefix="qwen_omni_placeholder_", suffix=".png")
    os.close(file_descriptor)
    image_path = Path(output_path)
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(image_path)
    return image_path.as_posix()


def run_qwen_top3_online(
    omni_url: str,
    timeout_sec: float,
    audio_file: str,
    candidate_classes: List[str],
    max_new_tokens: int,
    placeholder_image: str,
) -> Tuple[List[str], str]:
    candidate_text = ", ".join(candidate_classes)
    system_prompt = (
        "You are an audio classifier. "
        "Select the top 3 most likely classes from the candidate list. "
        "Do not use visual information."
    )
    user_prompt = (
        "Task: classify this audio clip into the top 3 most likely target categories.\n"
        f"Candidate categories: {candidate_text}\n"
        "Rules:\n"
        "1) Choose exactly three DISTINCT categories from the list.\n"
        "2) Sort them by likelihood from highest to lowest.\n"
        "3) Output format must be: category1, category2, category3\n"
        "4) Output ONLY three lowercase category tokens separated by commas."
    )
    payload = {
        "temperature": 0.1,
        "question": "",
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "video_path": placeholder_image,
        "audio_path": audio_file,
        "multi_video_path": "None",
        "history_actions": ["None"],
        "collision": False,
        "max_new_tokens": max(int(max_new_tokens), 1),
    }
    response = requests.post(omni_url, json=payload, timeout=float(timeout_sec))
    response.raise_for_status()
    raw_text = str(response.json().get("response", "")).strip()
    return parse_top3_from_text(raw_text, candidate_classes), raw_text


def build_label_classifier() -> torch.nn.Module:
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(512, 21)
    return model


def load_checkpoint_compat(checkpoint_path: str) -> Dict:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        if "Config" not in str(error):
            raise
        import habitat.config.default as habitat_default

        class Config(dict):
            pass

        habitat_default.Config = Config
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def extract_belief_state_dict(raw_checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if (
        isinstance(raw_checkpoint, dict)
        and "belief_predictor" in raw_checkpoint
        and isinstance(raw_checkpoint["belief_predictor"], dict)
    ):
        return raw_checkpoint["belief_predictor"]
    return raw_checkpoint


def load_label_weights_from_belief(
    model: torch.nn.Module, belief_state: Dict[str, torch.Tensor]
) -> None:
    cleaned = {
        key[len("classifier."):]: value
        for key, value in belief_state.items()
        if isinstance(key, str) and key.startswith("classifier.")
    }
    if not cleaned and "audiogoal_predictor" in belief_state:
        source = belief_state["audiogoal_predictor"]
        cleaned = {
            key[len("predictor."):]: value
            for key, value in source.items()
            if isinstance(key, str) and key.startswith("predictor.")
        }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"[WARN] label weights load: missing={len(missing)} unexpected={len(unexpected)}")


def build_qwen_prior(
    top3: List[str], classes_21: List[str], epsilon: float = 1e-6
) -> np.ndarray:
    prior = np.ones(len(classes_21), dtype=np.float32) * epsilon
    if not top3:
        return prior / prior.sum()
    weights = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    for index, category in enumerate(top3[:3]):
        if category in classes_21:
            prior[classes_21.index(category)] += weights[index]
    return prior / prior.sum()

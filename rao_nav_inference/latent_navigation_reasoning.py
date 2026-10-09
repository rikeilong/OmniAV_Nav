from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class LNRConfig:
    latent_tokens: int = 8
    optimization_steps: int = 4
    step_size: float = 0.1
    top_k: int = 4
    max_history: int = 20


def _load_frames(paths: Sequence[str], max_history: int) -> tuple[List[str], List[Image.Image]]:
    valid_paths: List[str] = []
    images: List[Image.Image] = []
    for raw_path in list(paths)[-max(int(max_history), 1):]:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            continue
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
        valid_paths.append(str(path.resolve()))
    return valid_paths, images


def _frame_embeddings(model, processor, images: Sequence[Image.Image]) -> torch.Tensor:
    image_inputs = processor.image_processor(images=list(images), return_tensors="pt")
    visual_device = next(model.visual.parameters()).device
    pixel_values = image_inputs["pixel_values"].to(visual_device)
    image_grid_thw = image_inputs["image_grid_thw"].to(visual_device)
    with torch.no_grad():
        features = model.get_image_features(pixel_values, image_grid_thw)
    merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    token_counts = (
        image_grid_thw.to(torch.long).prod(dim=-1) // (merge_size * merge_size)
    ).tolist()
    chunks = torch.split(features, [int(value) for value in token_counts], dim=0)
    return torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)


def _text_embeddings(model, processor, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = processor.tokenizer
    tokenized = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = tokenized.input_ids.to(model.device)
    embeddings = model.get_input_embeddings()(input_ids).squeeze(0)
    return input_ids.squeeze(0), embeddings


def select_lnr_frames(
    model,
    processor,
    history_image_paths: Sequence[str],
    predicted_audio_goal: str,
    config: LNRConfig,
) -> Dict[str, Any]:
    valid_paths, images = _load_frames(history_image_paths, config.max_history)
    if not valid_paths:
        return {"selected_paths": [], "selected_indices": [], "rewards": []}

    top_k = min(max(int(config.top_k), 1), len(valid_paths))
    if len(valid_paths) <= top_k:
        return {
            "selected_paths": valid_paths,
            "selected_indices": list(range(len(valid_paths))),
            "rewards": [],
        }

    frame_embeddings = _frame_embeddings(model, processor, images).to(model.device)
    frame_embeddings = frame_embeddings.to(model.dtype).detach()
    _, latent_seed_embeddings = _text_embeddings(model, processor, "<|latent|>")
    latent_seed = latent_seed_embeddings.mean(dim=0, keepdim=True)
    latent_tokens = latent_seed.repeat(max(int(config.latent_tokens), 1), 1).detach()
    latent_tokens = latent_tokens + torch.randn_like(latent_tokens) * 0.01
    goal_text = str(predicted_audio_goal).strip() or "unknown"
    goal_ids, goal_embeddings = _text_embeddings(model, processor, goal_text)
    _, prefix_embeddings = _text_embeddings(model, processor, "Predicted audio goal: ")
    rewards: List[float] = []

    for _ in range(max(int(config.optimization_steps), 1)):
        latent_tokens = latent_tokens.detach().requires_grad_(True)
        inputs_embeds = torch.cat(
            [frame_embeddings, latent_tokens, prefix_embeddings, goal_embeddings], dim=0
        ).unsqueeze(0)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        labels = torch.full(
            inputs_embeds.shape[:2], -100, dtype=torch.long, device=inputs_embeds.device
        )
        labels[0, -goal_ids.numel():] = goal_ids
        position_ids = torch.arange(
            inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
        ).view(1, 1, -1).expand(3, 1, -1)
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        reward = -outputs.loss
        gradient = torch.autograd.grad(reward, latent_tokens, retain_graph=False)[0]
        latent_tokens = latent_tokens + float(config.step_size) * gradient
        rewards.append(float(reward.detach().float().cpu()))

    query = latent_tokens.detach().mean(dim=0, keepdim=True)
    similarities = F.cosine_similarity(frame_embeddings.float(), query.float(), dim=-1)
    selected = torch.topk(similarities, k=top_k, largest=True).indices.tolist()
    selected = sorted(int(index) for index in selected)
    return {
        "selected_paths": [valid_paths[index] for index in selected],
        "selected_indices": selected,
        "similarities": [float(value) for value in similarities.detach().cpu()],
        "rewards": rewards,
    }

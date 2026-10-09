

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent



SOUND_SPACES_ROOT = Path(
    os.environ.get("SOUND_SPACES_ROOT", str(REPOSITORY_ROOT.parent))
).expanduser().resolve()


def _resolve_avn_data_root() -> Path:
    configured = os.environ.get("AVN_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (SOUND_SPACES_ROOT / "data").resolve()


AVN_DATA_ROOT = _resolve_avn_data_root()

# QWEN_MODEL_DIR = os.environ.get(
#     "QWEN_MODEL_DIR",
#     str(SOUND_SPACES_ROOT / "data/models/Qwen2.5-Omni-7B"),
# )

QWEN_MODEL_DIR = "/home/dataset-assist-0/model_checkpoints/avllm/Qwen2.5-Omni-7B"

SAVI_CONFIG = REPOSITORY_ROOT / "configs/experiments/savi.yaml"
TEST_CONTENT_DIR = AVN_DATA_ROOT / "datasets/semantic_audionav/mp3d/v1/test/content"
SAVI_BELIEF_CHECKPOINT = (
    AVN_DATA_ROOT / "pretrained_weights/semantic_audionav/savi/best_val.pth"
)
MP3D_SCENES_DIR = AVN_DATA_ROOT / "scene_datasets/mp3d"
BINAURAL_RIRS_DIR = AVN_DATA_ROOT / "binaural_rirs"
TEST_SOUNDS_DIR = AVN_DATA_ROOT / "sounds/semantic_splits/test"

LOCALIZATION_CHECKPOINT = Path(
    os.environ.get(
        "RAO_NAV_LOCALIZATION_CKPT",
        str(REPOSITORY_ROOT / "weights/localization_expert_qwen_best.pt"),
    )
).expanduser().resolve()
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results/navigation"

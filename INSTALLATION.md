# Installation and Evaluation

This guide installs RAO-Nav as an inference extension inside the official SoundSpaces repository. Two Conda environments are required because the SoundSpaces simulator and Qwen2.5-Omni use different dependency stacks.

## 1. Prerequisites

Recommended host configuration:

- Linux with an NVIDIA GPU and a compatible CUDA driver
- Conda or Miniconda
- Git, Git LFS, CMake, wget, curl, and ffmpeg
- Enough storage for Matterport3D assets and SoundSpaces audio data
- Access granted under the Matterport3D data-use terms

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs cmake wget curl ffmpeg
```

## 2. Install the official SoundSpaces environment

Follow the official SoundSpaces installation guide if its instructions differ from the commands below.

```bash
conda create -n ss python=3.9 cmake=3.14 -y
conda activate ss

mkdir -p /absolute/path/to
cd /absolute/path/to
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout RLRAudioPropagationUpdate
python setup.py install --headless --audio

cd /absolute/path/to
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.2
pip install -e .

cd /absolute/path/to
git clone https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
pip install -e .
```

Clone RAO-Nav inside SoundSpaces:

```bash
cd /absolute/path/to/sound-spaces
git clone https://github.com/<YOUR_GITHUB_ACCOUNT>/<YOUR_REPOSITORY>.git RAO-Nav-Inference-GitHub

conda activate ss
python -m pip install -r RAO-Nav-Inference-GitHub/requirements-ss.txt
```

The development environment used for this release has Python 3.9, PyTorch 2.5.1 with CUDA 12.1, torchvision 0.20.1, NumPy 1.26.4, and Gym 0.23.0. Install a PyTorch build compatible with your CUDA driver if SoundSpaces has not already installed one:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

## 3. Download SoundSpaces and MP3D data

From the SoundSpaces root, download the artifacts listed by the official `soundspaces/README.md`:

```bash
cd /absolute/path/to/sound-spaces
mkdir -p data
cd data

wget https://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs.tar
wget https://dl.fbaipublicfiles.com/SoundSpaces/metadata.tar.xz
wget https://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz
wget https://dl.fbaipublicfiles.com/SoundSpaces/datasets.tar.xz
wget https://dl.fbaipublicfiles.com/SoundSpaces/pretrained_weights.tar.xz

tar -xf binaural_rirs.tar
tar -xf metadata.tar.xz
tar -xf sounds.tar.xz
tar -xf datasets.tar.xz
tar -xf pretrained_weights.tar.xz
```

Matterport3D scene meshes cannot be redistributed with this repository. Request access and download MP3D according to the official Matterport3D and SoundSpaces instructions. Arrange the data so the following paths exist:

```text
sound-spaces/data/
├── binaural_rirs/mp3d/
├── datasets/semantic_audionav/mp3d/v1/test/content/
├── scene_datasets/mp3d/<SCENE_ID>/<SCENE_ID>.glb
├── scene_datasets/mp3d/<SCENE_ID>/<SCENE_ID>.navmesh
├── scene_datasets/mp3d/<SCENE_ID>/<SCENE_ID>.house
├── scene_datasets/mp3d/<SCENE_ID>/<SCENE_ID>_semantic.ply
└── sounds/
```

Check the test split before evaluation:

```bash
find /absolute/path/to/sound-spaces/data/datasets/semantic_audionav/mp3d/v1/test/content \
  -maxdepth 1 -type f | head
find /absolute/path/to/sound-spaces/data/scene_datasets/mp3d \
  -name '*.glb' | head
```

## 4. Create the Qwen2.5-Omni environment

```bash
conda create -n rao-qwen python=3.10 -y
conda activate rao-qwen

python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r \
  /absolute/path/to/sound-spaces/RAO-Nav-Inference-GitHub/requirements-omni.txt
```

The tested Qwen environment uses Python 3.10, PyTorch 2.6.0 with CUDA 12.4, Transformers 4.57.0, qwen-omni-utils 0.0.9, FastAPI 0.115.11, and Uvicorn 0.34.0. Change the PyTorch wheel only when required by your driver.

Download the official Qwen2.5-Omni-7B model:

```bash
conda activate rao-qwen
python -m pip install 'huggingface_hub[cli]'
mkdir -p /absolute/path/to/sound-spaces/data/models
hf download Qwen/Qwen2.5-Omni-7B \
  --local-dir /absolute/path/to/sound-spaces/data/models/Qwen2.5-Omni-7B
```

## 5. Configure local paths

Set environment variables in both terminals, or add them to your shell profile:

```bash
export SOUND_SPACES_ROOT=/absolute/path/to/sound-spaces
export AVN_DATA_ROOT="$SOUND_SPACES_ROOT/data"
export RAO_NAV_LOCALIZATION_CKPT="$SOUND_SPACES_ROOT/RAO-Nav-Inference-GitHub/weights/localization_expert_qwen_best.pt"
export QWEN_MODEL_DIR="$SOUND_SPACES_ROOT/data/models/Qwen2.5-Omni-7B"
export QWEN_OMNI_URL=http://127.0.0.1:6006/v1/omni/inference
```

Validate the resolved paths:

```bash
conda activate ss
cd /absolute/path/to/sound-spaces/RAO-Nav-Inference-GitHub
python - <<'PY'
from rao_nav_inference import paths
for name in (
    "SOUND_SPACES_ROOT",
    "AVN_DATA_ROOT",
    "TEST_CONTENT_DIR",
    "LOCALIZATION_CHECKPOINT",
    "QWEN_MODEL_DIR",
):
    print(name, getattr(paths, name))
PY
```

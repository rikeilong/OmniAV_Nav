#!/usr/bin/env sh
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SOUND_SPACES_ROOT=${SOUND_SPACES_ROOT:-$(dirname "$REPO_ROOT")}
NAV_PYTHON=${NAV_PYTHON:-python}
NAV_CUDA_VISIBLE_DEVICES=${NAV_CUDA_VISIBLE_DEVICES:-2}
QWEN_OMNI_URL=${QWEN_OMNI_URL:-http://127.0.0.1:6006/v1/omni/inference}

if [ -z "${AVN_DATA_ROOT:-}" ]; then
  if [ -d "$SOUND_SPACES_ROOT/data/datasets/semantic_audionav/mp3d/v1/test/content" ]; then
    AVN_DATA_ROOT="$SOUND_SPACES_ROOT/data"
  else
    echo "AVN_DATA_ROOT is required and no SemanticAudioNav test dataset was detected." >&2
    exit 1
  fi
fi

export SOUND_SPACES_ROOT
export AVN_DATA_ROOT
export PYTHONPATH="$REPO_ROOT:$SOUND_SPACES_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$NAV_CUDA_VISIBLE_DEVICES"
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/rao_nav_numba_cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/rao_nav_mpl_cache}

cd "$SOUND_SPACES_ROOT"
exec "$NAV_PYTHON" -m rao_nav_inference.run_qwen_omni_nav_iterative_mp3d_eval \
  --split test \
  --num-episodes 1000 \
  --max-steps 200 \
  --max-iterations 20 \
  --omni-url "$QWEN_OMNI_URL" \
  --request-timeout 120 \
  --localization-device cpu \
  --seed 0 \
  --save-dir "$REPO_ROOT/results/test1000" \
  "$@"

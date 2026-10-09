from __future__ import annotations

import os
from pathlib import Path

from ss_baselines.savi.config.default import get_config as _get_config

from rao_nav_inference.paths import REPOSITORY_ROOT


def get_config(config_paths=None, opts=None, model_dir=None, run_type=None, overwrite=False):
    if isinstance(config_paths, str) and "," not in config_paths:
        config_paths = str(Path(config_paths).expanduser().resolve())
    previous_directory = Path.cwd()
    try:
        os.chdir(REPOSITORY_ROOT)
        return _get_config(
            config_paths=config_paths,
            opts=opts,
            model_dir=model_dir,
            run_type=run_type,
            overwrite=overwrite,
        )
    finally:
        os.chdir(previous_directory)

import random
import subprocess
from pathlib import Path

import numpy as np


def create_parents(path: Path):
    folder = path.parent
    folder.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit_hash

    # Handle the case where the git command fails or isn't available for some reason
    except subprocess.CalledProcessError:
        return None

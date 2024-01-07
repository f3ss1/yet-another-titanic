import random
from pathlib import Path

import numpy as np


def create_parents(path: Path):
    folder = path.parent
    folder.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

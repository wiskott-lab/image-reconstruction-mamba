import os
from pathlib import Path
import torch

DATA_PATH = Path(os.getenv('DATA_PATH'))
RUNS_PATH = DATA_PATH / 'runs'
ANALYTICS = DATA_PATH / 'analytics'
DATASETS_PATH = Path(os.getenv('DATASETS_PATH'))
NEPTUNE_MODE = os.getenv('NEPTUNE_MODE', 'async')
RUNS_PATH.mkdir(parents=True, exist_ok=True)
ANALYTICS.mkdir(parents=True, exist_ok=True)
DATASETS_PATH.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

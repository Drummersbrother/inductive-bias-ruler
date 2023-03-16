"""
Defining constants and paths that are used across the project
"""
import os
from os import environ as env
from pathlib import Path

from dotenv import load_dotenv

BASE: Path = Path(Path.resolve(Path(os.path.join(__file__, "..", ".."))))
DATA: Path = BASE / "data"
MODELS: Path = DATA / "models"
DATASETS: Path = DATA / "datasets"
OUTPUTS: Path = DATA / "outputs"
PL_LOGS: Path = DATA / "pl_logs"
NEPTUNE_LOGS: Path = BASE / ".neptune"
DOTENV: Path = BASE / ".env"

load_dotenv(DOTENV)

assert "NEPTUNE_API_TOKEN" in env, "$NEPTUNE_API_TOKEN is not set, please modify either .env or your system-level env vars"

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from src import datasets, main, ml_logging, models, plotting, trainer, utils

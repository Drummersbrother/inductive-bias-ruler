"""
Defining constants and paths that are used across the project
"""
import os
from pathlib import Path

BASE: Path = Path(Path.resolve(Path(os.path.join(__file__, "..", ".."))))
DATA: Path = BASE / "data"
MODELS: Path = DATA / "models"
DATASETS: Path = DATA / "datasets"
OUTPUTS: Path = DATA / "outputs"
PL_LOGS: Path = DATA / "pl_logs"
NEPTUNE_LOGS: Path = BASE / ".neptune"
DOTENV: Path = BASE / ".env"

from dotenv import load_dotenv

load_dotenv(DOTENV)

from os import environ as env

assert "NEPTUNE_API_TOKEN" in env, "$NEPTUNE_API_TOKEN is not set, please modify either .env or your system-level env vars"

from src import models, datasets, plotting, utils, trainer, ml_logging

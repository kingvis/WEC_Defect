"""Shared utilities: reproducible seeds, logging, config loading, label maps."""
from __future__ import annotations
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Repo root = parent of the src/ directory.
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_logger(name: str = "wec") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_faults(path: str | Path = CONFIG_DIR / "faults.yaml") -> Dict[str, Any]:
    return load_yaml(path)


def label_map(path: str | Path = CONFIG_DIR / "faults.yaml") -> Dict[str, int]:
    """fault-name -> integer label, including enabled compound faults.

    Built from config/faults.yaml so labels stay data-driven (build guide §5.4).
    """
    cfg = load_faults(path)
    lm: Dict[str, int] = {name: spec["label"] for name, spec in cfg.get("faults", {}).items()
                          if spec.get("enabled", True)}
    for comp in cfg.get("compound_faults", []) or []:
        if comp.get("enabled", True):
            lm[comp["name"]] = comp["label"]
    return lm


def label_names(path: str | Path = CONFIG_DIR / "faults.yaml") -> Dict[int, str]:
    """integer label -> fault-name (inverse of label_map)."""
    return {v: k for k, v in label_map(path).items()}


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, OUTPUTS):
        d.mkdir(parents=True, exist_ok=True)


def contiguous_labels(y):
    """Map possibly-sparse semantic labels (e.g. 0,1,2,4) to contiguous model indices (0,1,2,3).

    Returns (y_contig, present) where present[i] is the original semantic label for model index i.
    Models need contiguous classes; the data card keeps the original semantic labels.
    """
    present = sorted(int(v) for v in np.unique(y))
    remap = {orig: i for i, orig in enumerate(present)}
    y_contig = np.asarray([remap[int(v)] for v in y])
    return y_contig, present

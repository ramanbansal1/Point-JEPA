from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from encoder.models import DualEncoder
from gen_final.model import TokenToPoint
import os

logging.getLogger("trimesh").setLevel(logging.ERROR)


# ==================================================
# Train config (FAST, PRETRAINED ENCODER)
# ==================================================
@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01

    max_iters: int = 30
    batch_size: int = 8
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    log_every: int = 5

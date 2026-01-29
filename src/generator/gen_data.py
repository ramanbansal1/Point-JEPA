from dataclasses import dataclass
from typing import Tuple, Literal
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh

# ==================================================
# Config (MINIMAL, SINGLE-MASK, LIDAR-STYLE)
# ==================================================
@dataclass
class ModelNetConfig:
    root: str = "../data"
    split: Literal["train", "val", "test"] = "train"

    # canonicalization
    center: bool = True
    scale: bool = True

    # geometry
    num_points_total: int = 7072          # oversampled surface
    num_context_points: int = 1024        # visible (LiDAR-like)
    num_target_points: int = 6048         # missing region (GT only)
    oversample_factor: int = 4
    min_points: int = 64

    # tokens (encoder / predictor side)
    token_points: int = 32
    token_dim: int = 1024

    # partial view
    partial_view_keep_ratio: float = 0.6

    # augmentation (rigid only)
    rotate: bool = True
    rotate_axis: Literal["z", "so3"] = "z"
    translate_std: float = 0.01
    scale_range: Tuple[float, float] = (0.9, 1.1)


# ==================================================
# Utilities
# ==================================================

def normalize_pc(pc: np.ndarray) -> np.ndarray:
    pc = pc - pc.mean(axis=0)
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc = pc / scale
    return pc


# ==================================================
# Deterministic spherical half-space mask (SINGLE MASK)
# ==================================================

def generate_spherical_directions(
    num_phi=8,
    num_theta=2,
    theta_range=(-15, 15),
    r=1.75,
):
    phis = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)
    theta_offsets = np.linspace(
        np.deg2rad(theta_range[0]),
        np.deg2rad(theta_range[1]),
        num_theta,
    )
    thetas = np.pi / 2 + theta_offsets

    dirs = []
    for theta in thetas:
        for phi in phis:
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            v = np.array([x, y, z])
            v = v / np.linalg.norm(v)
            dirs.append(v)

    return np.stack(dirs)


DIRECTIONS = generate_spherical_directions(num_phi=8, num_theta=2)


def sample_halfspace_mask(pc: np.ndarray, dir_id: int, min_points=64):
    """
    Single deterministic half-space mask.
    """
    u = DIRECTIONS[dir_id % len(DIRECTIONS)]
    proj = pc @ u
    thresh = np.median(proj)

    mask = proj <= thresh
    if mask.sum() < min_points:
        return None, None

    center = pc[mask].mean(axis=0)
    return mask, center


# ==================================================
# Dataset (SINGLE MASK, CONTEXT + TARGET)
# ==================================================
class ModelNetDataset(Dataset):
    def __init__(self, cfg: ModelNetConfig, samples_per_class: int = 100):
        self.cfg = cfg
        self.samples_per_class = samples_per_class

        self.allowed_classes = {
            "bed",
            "chair",
            "desk",
            "table",
            "bookshelf",
        }

        # collect all mesh paths per class
        self.files_by_class = {}

        for cls in sorted(os.listdir(cfg.root)):
            if cls not in self.allowed_classes:
                continue

            cls_path = os.path.join(cfg.root, cls, cfg.split)
            if not os.path.isdir(cls_path):
                continue

            files = [
                os.path.join(cls_path, f)
                for f in os.listdir(cls_path)
                if f.endswith(".off")
            ]

            if len(files) == 0:
                continue

            self.files_by_class[cls] = sorted(files)

        assert len(self.files_by_class) == len(self.allowed_classes), \
            "Some required classes are missing from the dataset."

        # active subset used by DataLoader
        self.mesh_paths = []
        self.resample_subset()

    def resample_subset(self):
        """
        Sample exactly `samples_per_class` meshes per class.
        Call once per epoch if desired.
        """
        self.mesh_paths = []

        for cls, files in self.files_by_class.items():
            k = min(self.samples_per_class, len(files))
            chosen = np.random.choice(files, size=k, replace=False)
            self.mesh_paths.extend(chosen)

        np.random.shuffle(self.mesh_paths)

    def __len__(self):
        return len(self.mesh_paths)  # = 5 * 100 = 500

    def load_mesh(self, path):
        mesh = trimesh.load(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump().sum()
        return mesh

    def sample_points(self, mesh):
        pts, _ = trimesh.sample.sample_surface_even(
            mesh,
            self.cfg.num_points_total * self.cfg.oversample_factor,
        )
        return pts

    def fixed_sample(self, pc: np.ndarray, n: int) -> np.ndarray:
        if len(pc) >= n:
            idx = np.random.choice(len(pc), n, replace=False)
            return pc[idx]
        else:
            pad = pc[np.random.choice(len(pc), n - len(pc), replace=True)]
            return np.concatenate([pc, pad], axis=0)

    def __getitem__(self, idx):
        mesh = self.load_mesh(self.mesh_paths[idx])
        pc = normalize_pc(self.sample_points(mesh))

        # deterministic direction per index
        dir_id = idx % len(DIRECTIONS)

        mask, center = sample_halfspace_mask(
            pc,
            dir_id=dir_id,
            min_points=self.cfg.min_points,
        )

        if mask is None:
            perm = np.random.permutation(len(pc))
            ctx = pc[perm[: self.cfg.num_context_points]]
            tgt = pc[perm[self.cfg.num_context_points :]]
        else:
            tgt = pc[mask]
            ctx = pc[~mask]

        ctx = self.fixed_sample(ctx, self.cfg.num_context_points)
        tgt = self.fixed_sample(tgt, self.cfg.num_target_points)

        return {
            "context_xyz": torch.from_numpy(ctx).float(),
            "target_xyz": torch.from_numpy(tgt).float(),
            "mask_center": torch.from_numpy(center).float(),
            "dir_id": dir_id,
        }

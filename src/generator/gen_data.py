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
    root: str = "data"
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
    def __init__(self, cfg: ModelNetConfig):
        self.cfg = cfg
        self.mesh_paths = []

        for cls in sorted(os.listdir(cfg.root)):
            cls_path = os.path.join(cfg.root, cls, cfg.split)
            if not os.path.isdir(cls_path):
                continue
            for f in os.listdir(cls_path):
                if f.endswith(".off"):
                    self.mesh_paths.append(os.path.join(cls_path, f))

        assert len(self.mesh_paths) > 0

    def __len__(self):
        return len(self.mesh_paths)

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
            # fallback: random split
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

# ==================================================
# Visualization Utilities (Context / Target / Mask)
# ==================================================
import random

def _rand_color():
    return np.array([
        random.uniform(0.2, 1.0),
        random.uniform(0.2, 1.0),
        random.uniform(0.2, 1.0),
        1.0,
    ])


def points_to_trimesh(pc, color):
    cloud = trimesh.points.PointCloud(pc)
    cloud.colors = np.tile((color * 255).astype(np.uint8), (len(pc), 1))
    return cloud


def sphere_at(center, radius=0.03, color=(255, 0, 0, 255)):
    sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
    sphere.apply_translation(center)
    sphere.visual.vertex_colors = np.tile(color, (len(sphere.vertices), 1))
    return sphere


def visualize_sample(sample):
    """
    sample from ModelNetDataset
    {
        context_xyz : [Nc, 3]
        target_xyz  : [Nt, 3]
        mask_center : [3]
        dir_id      : int
    }
    """
    scene = trimesh.Scene()

    ctx = sample["context_xyz"]
    tgt = sample["target_xyz"]
    center = sample["mask_center"]
    print(len(tgt))

    if torch.is_tensor(ctx): ctx = ctx.cpu().numpy()
    if torch.is_tensor(tgt): tgt = tgt.cpu().numpy()
    if torch.is_tensor(center): center = center.cpu().numpy()

    scene.add_geometry(points_to_trimesh(ctx, np.array([0.7, 0.7, 0.7, 1.0])))
    scene.add_geometry(points_to_trimesh(tgt, _rand_color()))
    scene.add_geometry(sphere_at(center))

    scene.show()

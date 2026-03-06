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
    num_points_total: int = 7072
    num_context_points: int = 6048
    num_target_points: int = 1024
    oversample_factor: int = 4
    min_points: int = 64

    # tokens
    token_points: int = 32
    token_dim: int = 1024

    # augmentation
    rotate: bool = True
    rotate_axis: Literal["z", "so3"] = "z"
    translate_std: float = 0.01
    scale_range: Tuple[float, float] = (0.9, 1.1)

# ==================================================
# Mask policy (LOCKED, LOW-ENTROPY)
# ==================================================
MASK_PROBS = {"box": 1.0}

# ==================================================
# Utilities
# ==================================================

def normalize_pc(pc: np.ndarray) -> np.ndarray:
    pc = pc.astype(np.float32)
    pc = pc - pc.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pc, axis=1))
    return pc / scale


def sample_box_mask(
    pc,
    center,
    scale_range=(0.25, 0.50),
    aspect_ratio_range=(0.8, 1.2),
):
    pc = pc.astype(np.float32)
    center = center.astype(np.float32)

    obj_scale = np.max(np.linalg.norm(pc, axis=1))
    scale = float(np.random.uniform(*scale_range)) * obj_scale
    aspect = float(np.random.uniform(*aspect_ratio_range))

    half_sizes = np.array([scale, scale, scale * aspect], dtype=np.float32)
    lower = center - half_sizes
    upper = center + half_sizes

    return np.all((pc >= lower) & (pc <= upper), axis=1)


def sample_random_mask(pc, min_points=64):
    """Single, low-entropy mask per sample."""
    N = len(pc)

    for _ in range(10):
        center = pc[np.random.randint(N)]
        mask = sample_box_mask(pc, center)
        ratio = mask.sum() / N

        if mask.sum() >= min_points and 0.10 <= ratio <= 0.20:
            center = pc[mask].mean(axis=0).astype(np.float32)
            return mask, center, "box"

    # fallback (rare)
    idx = np.random.randint(N)
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True
    return mask, pc[idx].astype(np.float32), "fallback"


def rigid_transform(pc, R, scale):
    pc = pc.astype(np.float32)
    R = R.astype(np.float32)
    scale = np.float32(scale)
    return (pc @ R.T) * scale


def add_noise(pc, noise_std=0.002):
    pc = pc.astype(np.float32)
    return pc + np.random.randn(*pc.shape).astype(np.float32) * noise_std

# ==================================================
# Dataset (SINGLE MASK, CACHED PER INDEX)
# ==================================================
class ModelNetDataset(Dataset):
    def __init__(self, cfg: ModelNetConfig, samples_per_class: int = 100):
        self.cfg = cfg
        self.samples_per_class = samples_per_class

        self.allowed_classes = {"bathtub"}
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
            if files:
                self.files_by_class[cls] = sorted(files)

        assert len(self.files_by_class) == len(self.allowed_classes)

        self.mesh_paths = []
        self.resample_subset()

        # cache masks per index
        self._mask_cache = {}

    def resample_subset(self):
        self.mesh_paths = []
        for cls, files in self.files_by_class.items():
            k = min(self.samples_per_class, len(files))
            chosen = np.random.choice(files, size=k, replace=False)
            self.mesh_paths.extend(chosen)
        np.random.shuffle(self.mesh_paths)

    def __len__(self):
        return len(self.mesh_paths)

    def load_mesh(self, path):
        mesh = trimesh.load(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump().sum()
        return mesh

    def sample_points(self, mesh):
        pts, _ = trimesh.sample.sample_surface_even(
            mesh, self.cfg.num_points_total * self.cfg.oversample_factor
        )
        return pts.astype(np.float32)

    def fixed_sample(self, pc: np.ndarray, n: int) -> np.ndarray:
        pc = pc.astype(np.float32)
        if len(pc) >= n:
            return pc[np.random.choice(len(pc), n, replace=False)]
        pad = pc[np.random.choice(len(pc), n - len(pc), replace=True)]
        return np.concatenate([pc, pad], axis=0).astype(np.float32)

    def __getitem__(self, idx):
        mesh = self.load_mesh(self.mesh_paths[idx])
        pc = normalize_pc(self.sample_points(mesh))

        if idx not in self._mask_cache:
            self._mask_cache[idx] = sample_random_mask(
                pc, min_points=self.cfg.min_points
            )

        mask, center, _ = self._mask_cache[idx]

        tgt = pc[mask]
        ctx = pc[~mask]

        ctx = self.fixed_sample(ctx, self.cfg.num_context_points)
        tgt = self.fixed_sample(tgt, self.cfg.num_target_points)

        return {
            "context_xyz": torch.from_numpy(ctx).float(),
            "target_xyz": torch.from_numpy(tgt).float(),
            "mask_center": torch.from_numpy(center).float(),
            "dir_id": -1,
        }

# ==================================================
# Collate (SYMMETRIC AUGMENTATION, FLOAT32 SAFE)
# ==================================================

def random_rotation_matrix(axis="so3"):
    if axis == "z":
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    a, b, c = np.random.uniform(0, 2 * np.pi, size=3)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=np.float32)
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]], dtype=np.float32)
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def gen_collate_fn(batch, cfg: ModelNetConfig, augment=True):
    ctx_list, tgt_list, center_list, dir_ids = [], [], [], []

    for sample in batch:
        ctx = sample["context_xyz"].numpy().astype(np.float32)
        tgt = sample["target_xyz"].numpy().astype(np.float32)
        center = sample["mask_center"].numpy().astype(np.float32)

        if augment:
            R = random_rotation_matrix(cfg.rotate_axis)
            scale = np.float32(np.random.uniform(*cfg.scale_range))

            ctx = rigid_transform(ctx, R, scale)
            tgt = rigid_transform(tgt, R, scale)
            center = rigid_transform(center[None], R, scale)[0]

            ctx = add_noise(ctx, 0.002)
            tgt = add_noise(tgt, 0.001)

        ctx_list.append(torch.from_numpy(ctx).float())
        tgt_list.append(torch.from_numpy(tgt).float())
        center_list.append(torch.from_numpy(center).float())
        dir_ids.append(sample["dir_id"])

    return {
        "context_xyz": torch.stack(ctx_list),
        "target_xyz": torch.stack(tgt_list),
        "mask_center": torch.stack(center_list),
        "dir_id": torch.tensor(dir_ids),
    }

def dataset_sanity_check(dataset, n=10):
    for i in range(n):
        s = dataset[i]
        ctx, tgt = s["context_xyz"], s["target_xyz"]
        center = s["mask_center"]
        visualize_matplotlib(s)
        print(
            f"[{i}] ctx={ctx.shape}, tgt={tgt.shape}, "
            f"tgt_ratio={tgt.shape[0] / (ctx.shape[0] + tgt.shape[0]):.2f}, "
            f"center_norm={center.norm():.3f}"
        )

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_matplotlib(sample):
    ctx = sample["context_xyz"].numpy()
    tgt = sample["target_xyz"].numpy()
    c = sample["mask_center"].numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(ctx[:,0], ctx[:,1], ctx[:,2], s=1, c="blue")
    ax.scatter(tgt[:,0], tgt[:,1], tgt[:,2], s=1, c="green")
    ax.scatter(c[0], c[1], c[2], s=80, c="red")

    ax.set_axis_off()
    plt.show()


if __name__=='__main__':
    data_cfg = ModelNetConfig()
    dataset = ModelNetDataset(data_cfg)
    dataset_sanity_check(dataset, n=10)


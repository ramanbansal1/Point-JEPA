from dataclasses import dataclass
from typing import Tuple, Literal, List
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh


# ==========================================================
# Configuration
# ==========================================================

@dataclass
class ModelNetConfig:
    root: str = "../data"
    split: Literal["train", "val", "test"] = "train"

    # canonicalization
    normalize: bool = True

    # geometry
    num_points_total: int = 8000
    num_context_points: int = 6000
    num_target_points: int = 1024
    oversample_factor: int = 4
    min_target_points: int = 64

    # augmentation
    rotate: bool = True
    rotate_axis: Literal["z", "so3"] = "z"
    translate_std: float = 0.0
    scale_range: Tuple[float, float] = (0.9, 1.1)

    # dataset control
    allowed_classes: Tuple[str, ...] = ("bathtub",)
    samples_per_class: int = 100


# ==========================================================
# Utilities
# ==========================================================

def normalize_pc(pc: np.ndarray) -> np.ndarray:
    pc = pc.astype(np.float32)
    pc = pc - pc.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pc, axis=1)) + 1e-8
    return pc / scale


def random_rotation_matrix(axis="so3"):
    if axis == "z":
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]], dtype=np.float32)

    a, b, c = np.random.uniform(0, 2 * np.pi, size=3)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(a), -np.sin(a)],
                   [0, np.sin(a),  np.cos(a)]], dtype=np.float32)

    Ry = np.array([[ np.cos(b), 0, np.sin(b)],
                   [0,          1, 0],
                   [-np.sin(b), 0, np.cos(b)]], dtype=np.float32)

    Rz = np.array([[np.cos(c), -np.sin(c), 0],
                   [np.sin(c),  np.cos(c), 0],
                   [0,          0,         1]], dtype=np.float32)

    return Rz @ Ry @ Rx


def rigid_transform(pc, R, scale):
    return (pc @ R.T) * scale


def add_noise(pc, std):
    return pc + np.random.randn(*pc.shape).astype(np.float32) * std


def estimate_context_resolution(pc):
    """
    Average nearest neighbor distance.
    """
    dists = np.linalg.norm(
        pc[:, None, :] - pc[None, :, :], axis=-1
    )
    np.fill_diagonal(dists, 1e6)
    return np.float32(dists.min(axis=1).mean())


def sample_mask_box(pc, min_points=64):
    N = len(pc)

    for _ in range(10):
        center = pc[np.random.randint(N)]
        obj_scale = np.max(np.linalg.norm(pc, axis=1))
        scale = np.random.uniform(0.25, 0.5) * obj_scale

        half = np.array([scale, scale, scale], dtype=np.float32)
        lower = center - half
        upper = center + half

        mask = np.all((pc >= lower) & (pc <= upper), axis=1)
        ratio = mask.sum() / N

        if mask.sum() >= min_points and 0.10 <= ratio <= 0.20:
            center = pc[mask].mean(axis=0).astype(np.float32)
            return mask, center

    # fallback
    idx = np.random.randint(N)
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True
    return mask, pc[idx].astype(np.float32)


# ==========================================================
# Dataset
# ==========================================================

class ModelNetDataset(Dataset):

    def __init__(self, cfg: ModelNetConfig):
        self.cfg = cfg
        self.mesh_paths = self._collect_files()
        self._mask_cache = {}

    def _collect_files(self) -> List[str]:
        mesh_paths = []

        for cls in sorted(os.listdir(self.cfg.root)):
            if cls not in self.cfg.allowed_classes:
                continue

            split_path = os.path.join(self.cfg.root, cls, self.cfg.split)
            if not os.path.isdir(split_path):
                continue

            files = [
                os.path.join(split_path, f)
                for f in os.listdir(split_path)
                if f.endswith(".off")
            ]

            k = min(self.cfg.samples_per_class, len(files))
            chosen = np.random.choice(files, size=k, replace=False)
            mesh_paths.extend(chosen)

        np.random.shuffle(mesh_paths)
        return mesh_paths

    def __len__(self):
        return len(self.mesh_paths)

    def load_mesh(self, path):
        mesh = trimesh.load(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump().sum()
        return mesh

    def sample_surface(self, mesh):
        pts, _ = trimesh.sample.sample_surface_even(
            mesh,
            self.cfg.num_points_total * self.cfg.oversample_factor,
        )
        return pts.astype(np.float32)

    def fixed_sample(self, pc, n):
        if len(pc) >= n:
            idx = np.random.choice(len(pc), n, replace=False)
            return pc[idx]
        pad = pc[np.random.choice(len(pc), n - len(pc), replace=True)]
        return np.concatenate([pc, pad], axis=0)

    def __getitem__(self, idx):

        mesh = self.load_mesh(self.mesh_paths[idx])
        pc = self.sample_surface(mesh)

        if self.cfg.normalize:
            pc = normalize_pc(pc)

        mask, center = sample_mask_box(
            pc,
            min_points=self.cfg.min_target_points,
        )



        target_xyz = pc[mask]
        context_xyz = pc[~mask]

        context_xyz = self.fixed_sample(
            context_xyz,
            self.cfg.num_context_points,
        )
        target_xyz = self.fixed_sample(
            target_xyz,
            self.cfg.num_target_points,
        )

        # resolution estimate (used for adaptive sampling)
        context_scale = estimate_context_resolution(context_xyz)

        return {
            "context_xyz": torch.from_numpy(context_xyz).float(),
            "target_xyz": torch.from_numpy(target_xyz).float(),
            "mask_center": torch.from_numpy(center).float(),
            "context_scale": torch.tensor(context_scale).float(),
        }


# ==========================================================
# Collate Function
# ==========================================================

def collate_fn(batch, cfg: ModelNetConfig, augment=True):

    ctx_list, tgt_list, center_list, scale_list = [], [], [], []

    for sample in batch:

        ctx = sample["context_xyz"].numpy()
        tgt = sample["target_xyz"].numpy()
        center = sample["mask_center"].numpy()
        context_scale = sample["context_scale"].item()

        if augment and cfg.split == "train":

            R = random_rotation_matrix(cfg.rotate_axis)
            scale = np.random.uniform(*cfg.scale_range)

            ctx = rigid_transform(ctx, R, scale)
            tgt = rigid_transform(tgt, R, scale)
            center = rigid_transform(center[None], R, scale)[0]

            ctx = add_noise(ctx, 0.002)
            tgt = add_noise(tgt, 0.001)

        ctx_list.append(torch.from_numpy(ctx).float())
        tgt_list.append(torch.from_numpy(tgt).float())
        center_list.append(torch.from_numpy(center).float())
        scale_list.append(torch.tensor(context_scale).float())

    return {
        "context_xyz": torch.stack(ctx_list),
        "target_xyz": torch.stack(tgt_list),
        "mask_center": torch.stack(center_list),
        "context_scale": torch.stack(scale_list),
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


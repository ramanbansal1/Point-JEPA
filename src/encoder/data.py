from dataclasses import dataclass
from typing import Optional, Tuple
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
from sklearn.neighbors import NearestNeighbors
import logging
logging.getLogger("trimesh").setLevel(logging.ERROR)

@dataclass
class ModelNetConfig:
    # Data
    root: str = "../data"
    split: str = "train"

    # Sampling
    num_points: int = 3072
    oversample_factor: int = 4

    # JEPA masking
    use_jepa_masking: bool = True
    num_masks: int = 4

    # BOX mask geometry (only boxes)
    box_scale_range: Tuple[float, float] = (0.2, 0.35)
    box_aspect_ratio_range: Tuple[float, float] = (0.75, 1.5)
    min_points: int = 64

MASK_PROBS = {
    "box": 0.50,
    "ball": 0.20,
    "strip": 0.20,
    "half": 0.10,
}


def normalize_pc(pc):
    pc = pc - pc.mean(axis=0)
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc = pc / scale
    return pc


def sample_ball_mask(pc, radius_ratio_range=(0.15, 0.2)):
    """
    pc: (N, 3) normalized point cloud
    returns: mask (N,) boolean
    """
    # random center from point cloud
    center = pc[np.random.randint(len(pc))]

    # object scale ~ max distance from origin
    scale = np.max(np.linalg.norm(pc, axis=1))

    # random radius
    ratio = np.random.uniform(*radius_ratio_range)
    radius = ratio * scale

    dists = np.linalg.norm(pc - center, axis=1)
    mask = dists <= radius
    return mask

def sample_box_mask(
    pc,
    center,
    scale_range=(0.2, 0.35),
    aspect_ratio_range=(0.75, 1.5),
):
    """
    Axis-aligned anisotropic box
    """
    obj_scale = np.max(np.linalg.norm(pc, axis=1))

    scale = np.random.uniform(*scale_range) * obj_scale
    aspect = np.random.uniform(*aspect_ratio_range)

    half_sizes = np.array([
        scale,
        scale,
        scale * aspect
    ])

    lower = center - half_sizes
    upper = center + half_sizes

    return np.all((pc >= lower) & (pc <= upper), axis=1)

def sample_strip_masks(
    pc,
    available,
    num_strips=(2, 4),
    width_ratio=(0.03, 0.08),
    min_points=64,
    max_total_ratio=0.45,
):
    N = len(pc)
    obj_scale = np.max(np.linalg.norm(pc, axis=1))
    max_total = int(max_total_ratio * N)

    K = np.random.randint(num_strips[0], num_strips[1] + 1)

    u = np.random.randn(3)
    u /= np.linalg.norm(u)

    proj = pc @ u
    width = np.random.uniform(*width_ratio) * obj_scale

    masks, centers = [], []
    total_masked = 0

    for _ in range(K):
        candidates = np.where(available)[0]
        if len(candidates) == 0:
            break

        c_idx = np.random.choice(candidates)
        c_val = proj[c_idx]

        mask = np.abs(proj - c_val) <= width
        mask &= available

        if mask.sum() < min_points:
            continue

        if total_masked + mask.sum() > max_total:
            break

        masks.append(mask)
        centers.append(pc[c_idx])
        available[mask] = False
        total_masked += mask.sum()

    return masks, centers

def sample_halfspace_mask(pc, available, min_points=64):
    u = np.random.randn(3)
    u /= np.linalg.norm(u)

    proj = pc @ u
    thresh = np.median(proj[available])

    mask = proj <= thresh
    mask &= available

    if mask.sum() < min_points:
        return None, None

    center = pc[mask].mean(axis=0)
    return mask, center

def estimate_density(pc, k=16):
    """
    pc: (N, 3)
    returns: density score (N,) higher = denser
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(pc)
    dists, _ = nbrs.kneighbors(pc)
    # exclude self-distance at index 0
    mean_dist = dists[:, 1:].mean(axis=1)
    density = 1.0 / (mean_dist + 1e-8)
    return density

def generate_jepa_masks(
    pc,
    num_masks=4,
    box_scale_range=(0.2, 0.35),
    box_aspect_ratio_range=(0.75, 1.5),
    min_points=64,
    density_quantile=0.7,
    knn_k=16,
):
    """
    Non-overlapping, density-aware, mixed-type JEPA masks
    (box / ball / strip / half-space)

    GUARANTEES:
    - exactly `num_masks` masks
    - exactly `num_masks` centers
    - batch-safe for torch.stack
    """

    N = len(pc)
    available = np.ones(N, dtype=bool)

    masks, centers = [], []

    sub = np.random.choice(N, min(2000, N), replace=False)
    density_sub = estimate_density(pc[sub])
    dense_region = np.zeros(N, dtype=bool)
    dense_region[sub] = density_sub >= np.quantile(density_sub, density_quantile)

    while len(masks) < num_masks and available.any():
        remaining = num_masks - len(masks)

        mask_type = np.random.choice(
            list(MASK_PROBS.keys()),
            p=list(MASK_PROBS.values()),
        )


        # ---- BOX ----
        if mask_type == "box":
            candidates = np.where(available & dense_region)[0]
            if len(candidates) == 0:
                continue

            idx = np.random.choice(candidates)
            center = pc[idx]
            mask = sample_box_mask(
                pc,
                center,
                scale_range=box_scale_range,
                aspect_ratio_range=box_aspect_ratio_range,
            ) & available

            if mask.sum() >= min_points:
                masks.append(mask)
                centers.append(center)
                available[mask] = False

        # ---- BALL ----
        elif mask_type == "ball":
            mask = sample_ball_mask(pc) & available
            if mask.sum() >= min_points:
                masks.append(mask)
                centers.append(pc[mask].mean(axis=0))
                available[mask] = False

        # ---- STRIP (multi-strip, clipped) ----
        elif mask_type == "strip":
            strip_masks, strip_centers = sample_strip_masks(
                pc,
                available,
                min_points=min_points,
            )

            for m, c in zip(strip_masks, strip_centers):
                if len(masks) >= num_masks:
                    break
                masks.append(m)
                centers.append(c)
                available[m] = False

        # ---- HALF-SPACE ----
        elif mask_type == "half":
            mask, center = sample_halfspace_mask(
                pc,
                available,
                min_points=min_points,
            )
            if mask is not None:
                masks.append(mask)
                centers.append(center)
                available[mask] = False

    # Safety fallback (extremely rare)
    all_idx = np.arange(N)

    while len(masks) < num_masks:
        # if nothing is available, sample from full pc
        if available.any():
            idx = np.random.choice(np.where(available)[0])
        else:
            idx = np.random.choice(all_idx)

        mask = np.zeros(N, dtype=bool)
        mask[idx] = True

        masks.append(mask)
        centers.append(pc[idx])

        if available.any():
            available[idx] = False

    context_pc = pc[available]
    target_pcs = [pc[m] for m in masks]

    return context_pc, target_pcs, centers

class ModelNetDataset(Dataset):
    def __init__(self, config: ModelNetConfig, samples_per_class=100):
        self.cfg = config
        self.samples_per_class = samples_per_class

        self.allowed_classes = {
            "bed", "chair", "desk", "table", "bookshelf"
        }

        # store all files per class
        self.files_by_cat = {}

        for cls in sorted(os.listdir(config.root)):
            if cls not in self.allowed_classes:
                continue

            cls_path = os.path.join(config.root, cls, config.split)
            if not os.path.isdir(cls_path):
                continue

            files = [
                os.path.join(cls_path, f)
                for f in os.listdir(cls_path)
                if f.endswith(".off")
            ]

            if len(files) == 0:
                continue

            self.files_by_cat[cls] = sorted(files)

        assert len(self.files_by_cat) == len(self.allowed_classes)

        # active subset used by DataLoader
        self.mesh_paths = []
        self.resample_subset()

    def resample_subset(self):
        """
        Sample exactly N meshes per class.
        Call once per epoch.
        """
        self.mesh_paths = []

        for cls, files in self.files_by_cat.items():
            k = min(self.samples_per_class, len(files))
            chosen = np.random.choice(files, size=k, replace=False)
            self.mesh_paths.extend(chosen)

        random.shuffle(self.mesh_paths)

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
            self.cfg.num_points * self.cfg.oversample_factor
        )
        return pts

    def fixed_sample(self, pc, n):
        """Force fixed-size point cloud"""
        if len(pc) >= n:
            idx = np.random.choice(len(pc), n, replace=False)
            return pc[idx]
        else:
            pad = pc[np.random.choice(len(pc), n - len(pc), replace=True)]
            return np.concatenate([pc, pad], axis=0)

    def __getitem__(self, idx):
        mesh = self.load_mesh(self.mesh_paths[idx])
        pc = normalize_pc(self.sample_points(mesh))  # [N, 3]

        if not self.cfg.use_jepa_masking:
            pc = self.fixed_sample(pc, self.cfg.num_points)
            return torch.from_numpy(pc).float()

        # ---- JEPA masking ----
        context_pc, target_pcs, centers = generate_jepa_masks(
            pc,
            num_masks=self.cfg.num_masks,
            box_scale_range=self.cfg.box_scale_range,
            box_aspect_ratio_range=self.cfg.box_aspect_ratio_range,
            min_points=self.cfg.min_points,
        )

        # FIX: force fixed-size context
        # context_pc may be empty after masking
        if len(context_pc) == 0:
            # fallback: sample from original point cloud
            context_pc = pc
        else:
            context_pc = self.fixed_sample(context_pc, self.cfg.num_points)

        targets = []
        for t, c in zip(target_pcs, centers):
            targets.append({
                "pcd": torch.from_numpy(t).float(),          # [P_t, 3]
                "center": torch.from_numpy(c).float(),       # [3]
            })

        return {
            "context": torch.from_numpy(context_pc).float(),  # [Nc=2048, 3]
            "targets": targets                                # list length M
        }



def jepa_collate_fn(batch):
    contexts = []
    all_targets = []
    all_centers = []

    try:
        # infer expected context size from first valid sample
        expected_N = batch[0]["context"].shape[0]

        for sample in batch:
            ctx = sample["context"]

            # ---- HARD CHECK ----
            if ctx.shape[0] != expected_N:
                raise ValueError(
                    f"context size mismatch: {ctx.shape[0]} vs {expected_N}"
                )

            contexts.append(ctx)

            targets = sample["targets"]
            pcs = [t["pcd"] for t in targets]
            centers = [t["center"] for t in targets]

            all_targets.append(pcs)
            all_centers.append(torch.stack(centers))

        return {
            "context": torch.stack(contexts),         # [B, Nc, 3]
            "targets": all_targets,                   # list[B][M][Ni,3]
            "mask_centers": torch.stack(all_centers), # [B, M, 3]
        }

    except Exception as e:
        # ---- DROP THIS BATCH ----
        # returning None tells DataLoader to skip it
        print(f"[WARN] Dropping batch in collate_fn: {e}")
        return None


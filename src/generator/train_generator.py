from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from encoder.models import DualEncoder
from generator.gen_data import ModelNetDataset, ModelNetConfig, gen_collate_fn
from generator.test_model import PointGenerator
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


# ==================================================
# FPS-based Hierarchical Chamfer (FAST + STABLE)
# ==================================================



def chamfer_cdist(x, y):
    """
    x: [B, N, 3]
    y: [B, M, 3]
    """
    # [B, N, M]
    dists = torch.cdist(x, y, p=2) ** 2

    # pred -> gt
    min_xy = dists.min(dim=2)[0]   # [B, N]
    # gt -> pred
    min_yx = dists.min(dim=1)[0]   # [B, M]

    return min_xy.mean() + min_yx.mean()

def repulsion_loss(x, k=5, h=0.03):
    """
    x: [B, N, 3]
    Encourages uniform spacing
    """
    dist = torch.cdist(x, x)                     # [B, N, N]
    knn = dist.topk(k + 1, largest=False).values[:, :, 1:]
    return torch.exp(-knn / h).mean()

def variance_loss(x, min_var=0.01):
    """
    x: [B, N, 3]
    Encourages spatial spread, prevents centroid collapse
    """
    # per-sample variance over points
    var = x.var(dim=1)          # [B, 3]
    
    # penalize variance below threshold
    loss = F.relu(min_var - var).mean()
    return loss



# ==================================================
# Training loop
# ==================================================

def train(
    encoder: DualEncoder,
    generator: PointGenerator,
    dataset: ModelNetDataset,
    cfg: TrainConfig,
):
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device(cfg.device)
    # ----------------------------
    # Encoder: frozen teacher
    # ----------------------------
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ----------------------------
    # Generator: trainable
    # ----------------------------
    generator.to(device)
    generator.train()

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=(cfg.device == "cuda"),
        collate_fn=lambda b: gen_collate_fn(b, data_cfg, augment=True),
    )

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    step = 0

    while step < cfg.max_iters:
        dataset.resample_subset()
        pbar = tqdm(loader, desc="Training generator (FAST)")
        for batch in pbar:
            if step >= cfg.max_iters:
                break

            # ----------------------------
            # Batch unpack
            # ----------------------------
            ctx_xyz = batch["context_xyz"].to(device)   # [B, Nc, 3]
            tgt_xyz = batch["target_xyz"].to(device)    # [B, Nt, 3]
            mask_centers = batch["mask_center"].to(device)  # [B, 3]
            gt_xyz = torch.cat([ctx_xyz, tgt_xyz], dim=1)  # [B, Nc+Nt, 3]

            B, Nc, _ = ctx_xyz.shape
            Nt = tgt_xyz.shape[1]

            # ----------------------------
            # DualEncoder inputs (single mask)
            # ----------------------------
            mask_centers = mask_centers.unsqueeze(1)      # [B, 1, 3]
            xyz_targets = [[tgt_xyz[b]] for b in range(B)]

            # ----------------------------
            # Encode (teacher, no grad)
            # ----------------------------
            with torch.no_grad():
                enc_out = encoder(
                    xyz_context=ctx_xyz,
                    mask_centers=mask_centers,
                    xyz_targets=xyz_targets,
                    mode="train",
                )

            # ----------------------------
            # Generator forward
            # ----------------------------
            gen_xyz = generator(
                ctx_xyz=enc_out["ctx_xyz"],          # [B, 32, 3]
                ctx_tokens=enc_out["ctx_tokens"],    # [B, 32, 1024]
                pred_tokens=enc_out["pred_tokens"],  # [B, 1, 32, 1024]
                mask_centers=mask_centers,           # [B, 1, 3]
            )

            gen_xyz = gen_xyz.view(B, -1, 3)
            # split prediction
            if gen_xyz.shape[1] < Nc + Nt:
                continue

            pred_ctx = gen_xyz[:, :Nc]
            pred_tgt = gen_xyz[:, Nc:Nc + Nt]
            # ----------------------------
            # Losses (TARGET dominates)
            # ----------------------------

            loss_ctx = chamfer_cdist(pred_ctx, ctx_xyz)
            loss_tgt = chamfer_cdist(pred_tgt, tgt_xyz)
            loss = 2.0 * loss_ctx + loss_tgt

            loss += 0.1 * variance_loss(pred_tgt, min_var=.1)

            # ----------------------------
            # Backprop
            # ----------------------------
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            # ----------------------------
            # Logging
            # ----------------------------
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        step += 1
        if step % cfg.log_every == 0:
            print("""HEllo""")
            torch.save(
                {
                    "step": step,
                    "generator_state": generator.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                f"checkpoints/generator_step_{step}.pt",
            )
            if step % 5 == 0:  # visualize every 5 steps
                visualize_trimesh(
                    ctx_xyz[0],
                    tgt_xyz[0],
                    gen_xyz[0],
                )
    pbar.close()
    torch.save(
        {
            "step": step,
            "generator_state": generator.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        "checkpoints/generator_final.pt",
    )

import matplotlib.pyplot as plt

def visualize_step(
    ctx_xyz,
    tgt_xyz,
    pred_xyz,
    step,
    save_dir="vis",
    max_points=2048,
):
    """
    ctx_xyz : [Nc, 3]
    tgt_xyz : [Nt, 3]
    pred_xyz: [N, 3]
    """

    os.makedirs(save_dir, exist_ok=True)

    def subsample(x, n):
        if x.shape[0] > n:
            idx = torch.randperm(x.shape[0])[:n]
            return x[idx]
        return x

    ctx = subsample(ctx_xyz, max_points).cpu().numpy()
    tgt = subsample(tgt_xyz, max_points).cpu().numpy()
    pred = subsample(pred_xyz, max_points).cpu().detach().numpy()

    fig = plt.figure(figsize=(12, 4))

    for i, (pts, title, color) in enumerate([
        (ctx, "Context", "blue"),
        (tgt, "GT Target", "green"),
        (pred, "Prediction", "red"),
    ]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=color)
        ax.set_title(title)
        ax.axis("off")
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/step_{step:05d}.png", dpi=150)
    plt.close(fig)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# enable interactive mode
plt.ion()

_fig = None
_axes = None

def visualize_step_live(
    ctx_xyz,
    tgt_xyz,
    pred_xyz,
    step,
    max_points=2048,
):
    """
    Live matplotlib 3D visualization.
    Shows Context (blue), Target GT (green), Prediction (red).
    """

    global _fig, _axes

    def subsample(x, n):
        if x.shape[0] > n:
            idx = torch.randperm(x.shape[0])[:n]
            return x[idx]
        return x

    ctx = subsample(ctx_xyz, max_points).cpu().numpy()
    tgt = subsample(tgt_xyz, max_points).cpu().numpy()
    pred = subsample(pred_xyz, max_points).cpu().detach().numpy()


    if _fig is None:
        _fig = plt.figure(figsize=(12, 4))
        _axes = [
            _fig.add_subplot(1, 3, 1, projection="3d"),
            _fig.add_subplot(1, 3, 2, projection="3d"),
            _fig.add_subplot(1, 3, 3, projection="3d"),
        ]

    for ax in _axes:
        ax.cla()

    for ax, pts, title, color in zip(
        _axes,
        [ctx, tgt, pred],
        ["Context", "GT Target", f"Prediction (step {step})"],
        ["blue", "green", "red"],
    ):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=color)
        ax.set_title(title)
        ax.axis("off")
        ax.view_init(elev=20, azim=45)

    plt.draw()
    plt.pause(0.001)

import trimesh

def visualize_trimesh(ctx, tgt, pred):
    import trimesh
    import numpy as np

    scene = trimesh.Scene()

    ctx_np = ctx.cpu().numpy()
    tgt_np = tgt.cpu().numpy()
    pred_np = pred.cpu().detach().numpy()

    # ----------------------------
    # Point clouds
    # ----------------------------
    """scene.add_geometry(
        trimesh.points.PointCloud(ctx_np, colors=[0, 0, 255, 120])
    )
    scene.add_geometry(
        trimesh.points.PointCloud(tgt_np, colors=[0, 255, 0, 120])
    )"""
    scene.add_geometry(
        trimesh.points.PointCloud(pred_np, colors=[255, 0, 0, 120])
    )

    # ----------------------------
    # Prediction centroid (BLOB CENTER)
    # ----------------------------
    pred_center = pred_np.mean(axis=0)
    scene.add_geometry(
        trimesh.creation.icosphere(
            radius=0.03, transform=trimesh.transformations.translation_matrix(pred_center)
        ),
        node_name="pred_centroid",
    )

    # ----------------------------
    # Target centroid (REFERENCE)
    # ----------------------------
    tgt_center = tgt_np.mean(axis=0)
    scene.add_geometry(
        trimesh.creation.icosphere(
            radius=0.03, transform=trimesh.transformations.translation_matrix(tgt_center)
        ),
        node_name="tgt_centroid",
    )

    # ----------------------------
    # Axes at prediction centroid
    # ----------------------------
    axis = trimesh.creation.axis(origin_size=0.02, axis_length=0.15)
    axis.apply_translation(pred_center)
    scene.add_geometry(axis)

    scene.show()


# ==================================================
# Entry
# ==================================================
if __name__ == "__main__":
    data_cfg = ModelNetConfig()
    train_cfg = TrainConfig()

    dataset = ModelNetDataset(data_cfg)

    encoder = DualEncoder()
    ckpt = torch.load("checkpoints/jepa_step_9.pt", map_location="cpu")
    encoder.load_state_dict(ckpt, strict=False)

    generator = PointGenerator(token_dim=data_cfg.token_dim)

    train(encoder, generator, dataset, train_cfg)
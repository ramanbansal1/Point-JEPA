from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from pytorch3d.ops import knn_points
from encoder.models import DualEncoder
from generator.gen_data import ModelNetDataset, ModelNetConfig
from generator import PointGenerator
logging.getLogger("trimesh").setLevel(logging.ERROR)

# ==================================================
# Train config (FAST, PRETRAINED ENCODER)
# ==================================================
@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01

    max_iters: int = 20_000
    batch_size: int = 8
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    log_every: int = 50


# ==================================================
# FPS-based Hierarchical Chamfer (FAST + STABLE)
# ==================================================

from pytorch3d.ops import sample_farthest_points


def fps_one_sided_cd(pred: torch.Tensor, gt: torch.Tensor, n_samples: int):
    """
    One-sided Chamfer using FPS subsampling (pred -> gt)
    pred, gt : [N,3], [M,3]
    """
    if pred.numel() == 0 or gt.numel() == 0:
        return pred.new_tensor(0.0)

    pred = pred.unsqueeze(0)
    gt = gt.unsqueeze(0)

    # FPS subsample
    pred_fps, _ = sample_farthest_points(pred, K=min(n_samples, pred.shape[1]))
    gt_fps, _ = sample_farthest_points(gt, K=min(n_samples, gt.shape[1]))

    d, _, _ = knn_points(pred_fps, gt_fps, K=1)
    return d.mean()


def hierarchical_fps_cd(pred: torch.Tensor, gt: torch.Tensor, levels=(128, 256, 512)):
    """
    Multi-scale FPS Chamfer (coarse -> fine)
    """
    loss = 0.0
    for k in levels:
        loss += fps_one_sided_cd(pred, gt, n_samples=k)
    return loss / len(levels)


# ==================================================
# Training loop
# ==================================================

def train(
    encoder: DualEncoder,
    generator: PointGenerator,
    dataset: ModelNetDataset,
    cfg: TrainConfig,
):
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
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    step = 0
    pbar = tqdm(total=cfg.max_iters, desc="Training generator (FAST)")

    while step < cfg.max_iters:
        for batch in loader:
            if step >= cfg.max_iters:
                break

            # ----------------------------
            # Batch unpack
            # ----------------------------
            ctx_xyz = batch["context_xyz"].to(device)   # [B, Nc, 3]
            tgt_xyz = batch["target_xyz"].to(device)    # [B, Nt, 3]
            mask_centers = batch["mask_center"].to(device)  # [B, 3]

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
                enc_out["ctx_xyz"],
                enc_out["ctx_tokens"],
                enc_out["pred_tokens"],
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
            loss = 0.0
            for b in range(B):
                # target completion (FAST)
                # hierarchical target completion
                loss += hierarchical_fps_cd(pred_tgt[b], tgt_xyz[b])

                # light context regularization (coarse only)
                loss += 0.05 * fps_one_sided_cd(pred_ctx[b], ctx_xyz[b], n_samples=128)

            loss = loss / B

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
            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            step += 1
            pbar.update(1)

    pbar.close()


# ==================================================
# Entry
# ==================================================
if __name__ == "__main__":
    data_cfg = ModelNetConfig()
    train_cfg = TrainConfig()

    dataset = ModelNetDataset(data_cfg)

    encoder = DualEncoder()
    ckpt = torch.load("/home/raman/Desktop/Projects/Point-JEPA/checkpoints/jepa_step_4_.pt", map_location="cpu")
    encoder.load_state_dict(ckpt, strict=False)

    generator = PointGenerator(token_dim=data_cfg.token_dim)

    train(encoder, generator, dataset, train_cfg)

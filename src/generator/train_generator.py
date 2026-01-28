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
from generator.generator import PointGenerator

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


def chamfer_cdist(x, y):
    """
    x: [B, N, 3]
    y: [B, M, 3]
    """
    print(x.shape, y.shape)
    # [B, N, M]
    dists = torch.cdist(x, y, p=2) ** 2

    # pred -> gt
    min_xy = dists.min(dim=2)[0]   # [B, N]
    # gt -> pred
    min_yx = dists.min(dim=1)[0]   # [B, M]

    return min_xy.mean() + min_yx.mean()


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
        pin_memory=(cfg.device == "cuda"),
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
            print(gen_xyz.shape)
            print(Nc, Nt)
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
                loss += chamfer_cdist(pred_tgt[b], tgt_xyz[b])

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

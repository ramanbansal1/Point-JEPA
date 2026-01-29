from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from encoder.models import DualEncoder
from generator.gen_data import ModelNetDataset, ModelNetConfig
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
            loss = chamfer_cdist(pred_tgt, tgt_xyz) 

            # ----------------------------
            # Backprop
            # ----------------------------
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()

            if step % cfg.log_every == 0:
                torch.save(
                    {
                        "step": step,
                        "generator_state": generator.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "loss": loss.item(),
                    },
                    f"checkpoints/generator_step_{step}.pt",
                )

            # ----------------------------
            # Logging
            # ----------------------------
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        step += 1
    pbar.close()

    torch.save(
        {
            "step": step,
            "generator_state": generator.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        "checkpoints/generator_final.pt",
    )



# ==================================================
# Entry
# ==================================================
if __name__ == "__main__":
    data_cfg = ModelNetConfig()
    train_cfg = TrainConfig()

    dataset = ModelNetDataset(data_cfg)

    encoder = DualEncoder()
    ckpt = torch.load("../checkpoints/jepa_step_4_.pt", map_location="cpu")
    encoder.load_state_dict(ckpt, strict=False)

    generator = PointGenerator(token_dim=data_cfg.token_dim)

    train(encoder, generator, dataset, train_cfg)
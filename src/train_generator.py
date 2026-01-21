from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch3d.ops import sample_farthest_points
from tqdm import tqdm

from models import DualEncoder, RecursivePointGenerator
from data import ModelNetDataset, ModelNetConfig, jepa_collate_fn


@dataclass
class TrainConfig:
    # optimizer
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # training
    max_iters: int = 5_000
    batch_size: int = 4
    num_workers: int = 4
    device: str = "cpu"

    # loss
    lambda_pres: float = 1.0

    # logging
    log_every: int = 1

    # checkpoint
    pretrained_encoder_path: str = "checkpoints/jepa_step_9.pt"

    # ---- early stopping ----
    early_stop_patience: int = 300
    early_stop_min_delta: float = 1e-3



def chamfer_distance(x, y):
    """
    x: [B, N, 3]
    y: [B, M, 3]
    """
    dist = torch.cdist(x, y)
    return dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()

def preservation_loss(partial, full):
    """
    partial: [B, Np, 3]
    full:    [B, Nf, 3]
    """
    dist = torch.cdist(partial, full)
    return dist.min(dim=2)[0].mean()


def fps(x, k):
    if x.shape[1] == k:
        return x
    y, _ = sample_farthest_points(x, K=k)
    return y

def train_generator(
    encoder: DualEncoder,
    generator: RecursivePointGenerator,
    dataset,
    cfg: TrainConfig,
):
    device = torch.device(cfg.device)

    # ---- load & freeze encoder ----
    encoder.load_state_dict(
        torch.load(cfg.pretrained_encoder_path, map_location=device)
    )
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ---- generator ----
    generator.to(device)
    generator.train()

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=jepa_collate_fn,
    )

    step = 0
    pbar = tqdm(total=cfg.max_iters, desc="Generator training")
    
    best_loss = float("inf")
    no_improve_steps = 0

    while step < cfg.max_iters:
        for batch in loader:
            if batch is None:
                continue

            if step >= cfg.max_iters:
                break

            context = batch["context"].to(device)         # [B, Nc, 3]
            mask_centers = batch["mask_centers"].to(device)

            # ---- JEPA inference (frozen) ----
            with torch.no_grad():
                out = encoder(
                    xyz_context=context,
                    mask_centers=mask_centers,
                    xyz_targets=None,
                    mode="infer",
                )

                ctx_xyz = out["ctx_xyz"]                 # [B, 32, 3]
                pred_tokens = out["pred_tokens"]         # [B, M, 32, C]

            # ---- Generator ----
            final = generator(
                ctx_xyz,
                pred_tokens,
            )
            print(pred_tokens.shape)

            # ---- Build GT (context + targets) ----
            gt_full = []
            for b in range(context.shape[0]):
                pcs = [context[b]]
                for t in batch["targets"][b]:
                    pcs.append(t.to(device))
                gt_full.append(torch.cat(pcs, dim=0))

            # ---- Completion loss (multi-scale CD) ----
            L_completion = 0.0
            for b in range(context.shape[0]):
                pred = final[b].unsqueeze(0)          # [1, 3072, 3]
                gt   = gt_full[b].unsqueeze(0)        # [1, Ni, 3]

                k = min(pred.shape[1], gt.shape[1])
                pred_ds = fps(pred, k)
                gt_ds = fps(gt, k)
                L_completion += chamfer_distance(pred_ds, gt_ds)


            L_completion /= context.shape[0]

            # ---- Preservation loss ----
            L_pres = 0.0
            for b in range(context.shape[0]):
                L_pres += preservation_loss(
                    context[b:b+1],
                    final[b:b+1]
                )

            L_pres /= context.shape[0]

            loss = L_completion + cfg.lambda_pres * L_pres


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()  

            if current_loss < best_loss - cfg.early_stop_min_delta:
                best_loss = current_loss
                no_improve_steps = 0

                torch.save(
                    generator.state_dict(),
                    "checkpoints/generator_best.pt"
                )
            else:
                no_improve_steps += 1

            if no_improve_steps >= cfg.early_stop_patience:
                print(f"\nEarly stopping at step {step}")
                pbar.close()
                return


            if step % cfg.log_every == 0:
                pbar.set_postfix(
                    step=step,
                    loss=f"{loss.item():.4f}",
                    comp=f"{L_completion.item():.4f}",
                    pres=f"{L_pres.item():.4f}",
                )

            step += 1
            pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    data_cfg = ModelNetConfig()
    train_cfg = TrainConfig()

    dataset = ModelNetDataset(data_cfg)

    encoder = DualEncoder()
    generator = RecursivePointGenerator(
        upsample_factors=(6, 4, 4)  # 32 → 3072
    )

    train_generator(encoder, generator, dataset, train_cfg)

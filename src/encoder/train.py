from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from encoder.models import DualEncoder
from encoder.data import ModelNetDataset, ModelNetConfig, jepa_collate_fn
from tqdm import tqdm
from utils import save_checkpoint, knn_points
import logging
import matplotlib.pyplot as plt
logging.getLogger("trimesh").setLevel(logging.ERROR)

@dataclass
class TrainConfig:
    # optimizer
    lr: float = 1e-4                  # γ
    weight_decay: float = 0.1         # λ
    momentum: float = 0.95            # μ
    nesterov: bool = True
    ns_coefficients: Tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    ns_steps: int = 5
    eps: float = 1e-7

    # training
    max_iters: int = 10
    batch_size: int = 40
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else 'cpu'

    # EMA (JEPA)
    ema_decay: float = 0.99

    # early stopping (batch-level)
    early_stop_patience: int = 200
    early_stop_min_delta: float = 1e-4

    # logging
    log_every: int = 50


class BatchEarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, loss):
        if loss < self.best - self.min_delta:
            self.best = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def build_muon_optimizer(model, cfg: TrainConfig):
    return torch.optim.Muon(
        params=model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
        ns_coefficients=cfg.ns_coefficients,
        eps=cfg.eps,
        ns_steps=cfg.ns_steps,
        adjust_lr_fn=None,   # can be plugged later if needed
    )


def jepa_loss(pred_tokens, target_tokens, ctx_xyz, tgt_xyz):
    """
    Ragged JEPA loss (matches current DualEncoder.forward)
    """
    B, M, P_ctx, C = pred_tokens.shape
    loss = 0.0
    count = 0

    for b in range(B):
        ctx_pts = ctx_xyz[b]              # [P_ctx, 3]

        for m in range(M):
            tgt_pts = tgt_xyz[b][m]       # [P_t', 3]
            tgt_tok = target_tokens[b][m] # [P_t', C]
            pred_tok = pred_tokens[b, m]  # [P_ctx, C]

            # knn_points requires batch dimension
            _, idx, _ = knn_points(
                ctx_pts.unsqueeze(0),     # [1, P_ctx, 3]
                tgt_pts.unsqueeze(0),     # [1, P_t', 3]
                K=1,
            )

            idx = idx.squeeze(0).squeeze(-1)  # [P_ctx]

            aligned_target = tgt_tok[idx]     # [P_ctx, C]

            loss += F.mse_loss(pred_tok, aligned_target)
            count += 1

    return loss / count

@torch.no_grad()
def compute_pointwise_jepa_error(
    pred_tokens, target_tokens, ctx_xyz, tgt_xyz
):
    """
    Returns:
        xyz   : [P_ctx, 3]
        error : [P_ctx]
    Uses only first sample in batch and first mask (for visualization)
    """

    # take first batch element, first mask
    pred_tok = pred_tokens[0, 0]       # [P_ctx, C]
    tgt_tok  = target_tokens[0][0]     # [P_t', C]
    ctx_pts  = ctx_xyz[0]              # [P_ctx, 3]
    tgt_pts  = tgt_xyz[0][0]            # [P_t', 3]

    # align target tokens to context points
    _, idx, _ = knn_points(
        ctx_pts.unsqueeze(0),
        tgt_pts.unsqueeze(0),
        K=1,
    )
    idx = idx.squeeze(0).squeeze(-1)    # [P_ctx]

    aligned_tgt = tgt_tok[idx]          # [P_ctx, C]

    # L2 error per point
    error = (pred_tok - aligned_tgt).pow(2).mean(dim=1)

    return ctx_pts.detach().cpu(), error.detach().cpu()


def plot_context_with_error(xyz, error, step):
    """
    xyz   : [N, 3]
    error : [N]
    """

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=error,
        cmap="viridis",
        s=20,
    )

    fig.colorbar(p, ax=ax, shrink=0.6, label="JEPA token error")
    ax.set_title(f"Context XYZ + Token Error (step {step})")

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()



def train_jepa(
    model,
    dataset,
    cfg: TrainConfig,
):
    device = torch.device(cfg.device)
    model.to(device)
    model.train()



    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05
    )

    early_stopper = BatchEarlyStopping(
        cfg.early_stop_patience,
        cfg.early_stop_min_delta,
    )

    step = 0

    while step < cfg.max_iters:
        dataset.resample_subset()
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
            collate_fn=jepa_collate_fn,
        )
        pbar = tqdm(loader, desc=f"JEPA training", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            if step >= cfg.max_iters:
                break

            xyz_context = batch["context"].to(device)        # [B, Nc, 3]
            mask_centers = batch["mask_centers"].to(device) # [B, M, 3]

            # targets is now a list of length B
            xyz_targets = [
                [pcd.to(device) for pcd in sample_targets]
                for sample_targets in batch["targets"]
            ]

            out = model(
                xyz_context=xyz_context,
                mask_centers=mask_centers,
                xyz_targets=xyz_targets,
                mode="train",
            )

            loss = jepa_loss(
                pred_tokens=out["pred_tokens"],
                target_tokens=out["target_tokens"],
                ctx_xyz=out["ctx_xyz"],
                tgt_xyz=out["tgt_xyz"],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            model.update_target_encoder()

            if early_stopper.step(loss.item()):
                print(f"Early stopping at step {step}")
                return

            pbar.set_postfix(
                step=step,
                loss=f"{loss.item():.4f}"
            )


            save_checkpoint(model, step)
        # ---- Epoch-level visualization ----
        with torch.no_grad():
            xyz_vis, err_vis = compute_pointwise_jepa_error(
                pred_tokens=out["pred_tokens"],
                target_tokens=out["target_tokens"],
                ctx_xyz=out["ctx_xyz"],
                tgt_xyz=out["tgt_xyz"],
            )

        plot_context_with_error(xyz_vis, err_vis, step)


        step += 1


if __name__=='__main__':
    data_config = ModelNetConfig()
    train_cinfig = TrainConfig()
    dataset = ModelNetDataset(data_config)
    model = DualEncoder()
    state_dict = torch.load('checkpoints/jepa_step_9.pt')
    model.load_state_dict(state_dict)
    train_jepa(model, dataset, train_cinfig)
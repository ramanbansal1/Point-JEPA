import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_final.data import ModelNetConfig, ModelNetDataset, collate_fn
from gen_final.model import (
    sample_random_bounded,
    slice_mask,
    compute_pca_direction,
    knn_graph,
    SlicedGraphGenerator,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
import logging
logging.getLogger("trimesh").setLevel(logging.ERROR)

# -------------------------------------------------
# Chamfer Distance
# -------------------------------------------------

import trimesh
import numpy as np


def visualize_result(context_xyz, gt_full, sampled_xyz, refined_xyz):

    ctx = context_xyz[0].detach().cpu().numpy()
    gt = gt_full[0].detach().cpu().numpy()
    samp = sampled_xyz[0].detach().cpu().numpy()
    ref = refined_xyz[0].detach().cpu().numpy()

    pc_ctx = trimesh.points.PointCloud(ctx, colors=[0, 0, 255, 255])
    pc_gt = trimesh.points.PointCloud(gt, colors=[0, 255, 0, 255])
    pc_samp = trimesh.points.PointCloud(samp, colors=[255, 0, 0, 255])
    pc_ref = trimesh.points.PointCloud(ref, colors=[255, 255, 0, 255])

    scene = trimesh.Scene([pc_ctx, pc_gt, pc_samp, pc_ref])
    scene.show()

@torch.no_grad()
def visualize_model_output(model, loader):

    model.eval()

    batch = next(iter(loader))

    context_xyz = batch["context_xyz"].to(device)
    target_xyz = batch["target_xyz"].to(device)
    mask_center = batch["mask_center"].to(device)

    gt_full = torch.cat([context_xyz, target_xyz], dim=1)

    sampled_xyz = sample_random_bounded(
        context_xyz,
        mask_center,
        total_points=gt_full.shape[1],
    ).to(device)

    refined_xyz = model.forward_inference(sampled_xyz.clone())

    visualize_result(context_xyz, gt_full, sampled_xyz, refined_xyz)


def chamfer_distance(x, y):
    """
    x: [B, Nx, 3]
    y: [B, Ny, 3]
    """
    dist = torch.cdist(x, y)

    loss_xy = torch.mean(torch.min(dist, dim=2).values)
    loss_yx = torch.mean(torch.min(dist, dim=1).values)

    return loss_xy + loss_yx


# -------------------------------------------------
# Training Step
# -------------------------------------------------

def train_one_epoch(model, loader, optimizer):

    model.train()
    total_loss = 0

    for batch in tqdm(loader):

        context_xyz = batch["context_xyz"].to(device)
        target_xyz = batch["target_xyz"].to(device)
        mask_center = batch["mask_center"].to(device)

        # Combine GT full cloud
        gt_full = torch.cat([context_xyz, target_xyz], dim=1)

        # Random initialization
        sampled_xyz = sample_random_bounded(
            context_xyz,
            mask_center,
            total_points=gt_full.shape[1],
        )

        sampled_xyz = sampled_xyz.to(device)

        direction = compute_pca_direction(sampled_xyz)
        ts = torch.linspace(-1, 1, model.num_slices, device=device)

        loss = 0

        for t in ts:

            mask_src = slice_mask(sampled_xyz, direction, t, model.delta)
            mask_tgt = slice_mask(gt_full, direction, t, model.delta)

            for b in range(sampled_xyz.shape[0]):

                idx_src = mask_src[b]
                idx_tgt = mask_tgt[b]

                if idx_src.sum() < 5 or idx_tgt.sum() < 5:
                    continue

                slice_src = sampled_xyz[b][idx_src].unsqueeze(0)
                slice_tgt = gt_full[b][idx_tgt].unsqueeze(0)

                knn_idx = knn_graph(slice_src, model.k)

                delta = model.unet(slice_src, knn_idx)

                pred_slice = slice_src + delta

                loss += chamfer_distance(pred_slice, slice_tgt)

        loss = loss / model.num_slices

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    data_cfg = ModelNetConfig(root="../data", split="train")
    dataset = ModelNetDataset(data_cfg)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, data_cfg, augment=True),
    )

    model = SlicedGraphGenerator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50

    for epoch in range(epochs):

        loss = train_one_epoch(model, loader, optimizer)

        print(f"Epoch {epoch+1} | Loss: {loss:.6f}")
        if (epoch + 1) % 5 == 0:
            visualize_model_output(model, loader)

        torch.save(model.state_dict(), "sliced_model.pt")


if __name__ == "__main__":
    main()

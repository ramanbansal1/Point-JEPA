import torch
import numpy as np
import trimesh

from encoder.models import DualEncoder
from generator.gen_data import (
    ModelNetDataset,
    ModelNetConfig,
)
from generator.test_model import PointGenerator


# ----------------------------
# Chamfer-free inference
# ----------------------------
@torch.no_grad()
def run_inference(
    encoder,
    generator,
    sample,
    device="cuda",
):
    encoder.eval()
    generator.eval()

    ctx_xyz = sample["context_xyz"].unsqueeze(0).to(device)  # [1, Nc, 3]
    tgt_xyz = sample["target_xyz"].unsqueeze(0).to(device)   # [1, Nt, 3]
    mask_center = sample["mask_center"].unsqueeze(0).unsqueeze(1).to(device)

    # encoder expects ragged target list
    xyz_targets = [[tgt_xyz[0]]]

    enc_out = encoder(
        xyz_context=ctx_xyz,
        mask_centers=mask_center,
        xyz_targets=xyz_targets,
        mode="train",
    )

    gen_xyz = generator(
        enc_out["ctx_tokens"],
        enc_out["pred_tokens"],
    )

    gen_xyz = gen_xyz.view(1, -1, 3)

    Nc = ctx_xyz.shape[1]
    Nt = tgt_xyz.shape[1]

    pred_ctx = gen_xyz[:, :Nc]
    pred_tgt = gen_xyz[:, Nc:Nc + Nt]

    return (
        ctx_xyz.squeeze(0).cpu().numpy(),
        tgt_xyz.squeeze(0).cpu().numpy(),
        pred_tgt.squeeze(0).cpu().numpy(),
        sample["mask_center"].cpu().numpy(),
    )


# ----------------------------
# Trimesh visualization
# ----------------------------
def visualize_prediction(ctx, tgt_gt, tgt_pred, center):
    scene = trimesh.Scene()

    ctx_cloud = trimesh.points.PointCloud(
        ctx,
        colors=np.tile([180, 180, 180, 255], (len(ctx), 1)),
    )

    gt_cloud = trimesh.points.PointCloud(
        tgt_gt,
        colors=np.tile([0, 180, 0, 255], (len(tgt_gt), 1)),
    )

    pred_cloud = trimesh.points.PointCloud(
        tgt_pred,
        colors=np.tile([200, 30, 30, 255], (len(tgt_pred), 1)),
    )

    sphere = trimesh.creation.icosphere(radius=0.03)
    sphere.apply_translation(center)
    sphere.visual.vertex_colors = [255, 255, 0, 255]

    scene.add_geometry(ctx_cloud)
    scene.add_geometry(gt_cloud)
    scene.add_geometry(pred_cloud)
    scene.add_geometry(sphere)

    scene.show()


# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- dataset ----
    data_cfg = ModelNetConfig()
    dataset = ModelNetDataset(data_cfg)

    sample = dataset[np.random.randint(len(dataset))]

    # ---- models ----
    encoder = DualEncoder().to(device)
    generator = PointGenerator(token_dim=data_cfg.token_dim).to(device)

    encoder_ckpt = torch.load(
        "../checkpoints/jepa_step_4_.pt",
        map_location=device,
    )
    gen_ckpt = torch.load(
        "../checkpoints/generator_final.pt",
        map_location=device,
    )

    encoder.load_state_dict(encoder_ckpt, strict=False)
    generator.load_state_dict(gen_ckpt["generator_state"])

    # ---- inference ----
    ctx, tgt_gt, tgt_pred, center = run_inference(
        encoder,
        generator,
        sample,
        device=device,
    )

    # ---- visualize ----
    visualize_prediction(ctx, tgt_gt, tgt_pred, center)

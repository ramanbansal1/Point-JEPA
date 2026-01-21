import torch
import numpy as np
import trimesh

from data import ModelNetDataset, ModelNetConfig
from models import DualEncoder, RecursivePointGenerator


def visualize_generated(ctx_xyz, gen_xyz):
    """
    ctx_xyz : [32, 3]
    gen_xyz : [3072, 3]
    """
    scene = trimesh.Scene()

    # context anchors (red)
    ctx_cloud = trimesh.points.PointCloud(ctx_xyz)
    ctx_cloud.colors = np.tile(
        np.array([255, 0, 0, 255], dtype=np.uint8),
        (len(ctx_xyz), 1)
    )
    scene.add_geometry(ctx_cloud)

    # generated points (gray)
    gen_cloud = trimesh.points.PointCloud(gen_xyz)
    gen_cloud.colors = np.tile(
        np.array([180, 180, 180, 255], dtype=np.uint8),
        (len(gen_xyz), 1)
    )
    scene.add_geometry(gen_cloud)

    scene.show()


@torch.no_grad()
def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- dataset ----
    cfg = ModelNetConfig()
    dataset = ModelNetDataset(cfg)

    sample = dataset[10]

    context = sample["context"].unsqueeze(0).to(device)       # [1, Nc, 3]
    mask_centers = torch.stack(
        [t["center"] for t in sample["targets"]]
    ).unsqueeze(0).to(device)                                  # [1, M, 3]

    # ---- JEPA encoder ----
    jepa = DualEncoder().to(device)
    jepa.eval()

    out = jepa(
        xyz_context=context,
        mask_centers=mask_centers,
        xyz_targets=None,
        mode="infer",
    )

    ctx_xyz = out["ctx_xyz"]           # [1, 32, 3]
    pred_tokens = out["pred_tokens"]   # [1, M, 32, C]

    # ---- Generator ----
    generator = RecursivePointGenerator(
        upsample_factors=(6, 4, 4)
    ).to(device)
    generator.eval()

    gen_xyz = generator(ctx_xyz, pred_tokens)  # [1, 3072, 3]

    # ---- visualize ----
    visualize_generated(
        ctx_xyz[0].cpu().numpy(),
        gen_xyz[0].cpu().numpy(),
    )


if __name__ == "__main__":
    run_inference()

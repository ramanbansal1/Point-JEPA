import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, knn_graph
import numpy as np
import trimesh
from data import ModelNetDataset, ModelNetConfig
from models import DualEncoder
class XYZEdgeConv(nn.Module):

    """
    Correct XYZ-aware EdgeConv.
    EdgeConv handles message construction.
    """

    def __init__(self, feat_dim):
        super().__init__()

        in_dim = feat_dim + 3  # feat + xyz

        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        self.conv = EdgeConv(self.mlp, aggr="max")

    def forward(self, xyz, feat, edge_index):
        """
        xyz  : [N, 3]
        feat : [N, C]
        """
        node_feat = torch.cat([feat, xyz], dim=-1)  # [N, C+3]
        return self.conv(node_feat, edge_index)

class ContextUpsampler(nn.Module):
    """
    Non-trainable local densification around context points
    """

    def __init__(self, r_ctx=12, radius=0.02):
        super().__init__()
        self.r = r_ctx
        self.radius = radius

    def forward(self, ctx_xyz, ctx_feat):
        """
        ctx_xyz  : [B, P, 3]
        ctx_feat : [B, P, C]
        """
        B, P, _ = ctx_xyz.shape
        C = ctx_feat.shape[-1]

        # Repeat
        xyz = ctx_xyz.unsqueeze(2).repeat(1, 1, self.r, 1)
        feat = ctx_feat.unsqueeze(2).repeat(1, 1, self.r, 1)

        # Local jitter (non-trainable)
        noise = torch.randn(B, P, self.r, 3, device=ctx_xyz.device)
        noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-6)
        noise = noise * self.radius

        xyz = xyz + noise

        # Flatten
        xyz = xyz.view(B, P * self.r, 3)
        feat = feat.view(B, P * self.r, C)

        return xyz, feat

class PredUpsampler(nn.Module):
    """
    Non-trainable geometric seeding for missing regions
    """

    def __init__(self, r_pred=3, radius=0.05):
        super().__init__()
        self.r = r_pred
        self.radius = radius

    def forward(self, token, anchor_xyz):
        """
        token      : [B, C]
        anchor_xyz : [B, 3]
        """
        B, C = token.shape

        # Random isotropic seeds
        noise = torch.randn(B, self.r, 3, device=anchor_xyz.device)
        noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-6)
        noise = noise * self.radius

        xyz = anchor_xyz.unsqueeze(1) + noise       # [B, r, 3]
        feat = token.unsqueeze(1).repeat(1, self.r, 1)

        return xyz, feat


class OffsetMLP(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 3)
        )

    def forward(self, feat):
        return self.mlp(feat)

def apply_edgeconv_batched(edge_conv, xyz, feat, edge_index_fn):
    """
    xyz  : [B, N, 3]
    feat : [B, N, C]
    returns: [B, N, C]
    """
    B, N, _ = xyz.shape
    out = []

    for b in range(B):
        edge_index = edge_index_fn(xyz[b])  # per-batch graph
        feat_b = edge_conv(xyz[b], feat[b], edge_index)
        out.append(feat_b)

    return torch.stack(out, dim=0)


class JEPAPointDecoder(nn.Module):
    def __init__(self, feat_dim=128, up_r=12, pred_r=3):
        super().__init__()

        # Shared
        self.feat_proj = nn.Linear(1024, feat_dim)
        self.edge_conv = XYZEdgeConv(feat_dim)

        # Non-trainable upsamplers
        self.ctx_upsample = ContextUpsampler(r_ctx=up_r)
        self.pred_upsample = PredUpsampler(r_pred=pred_r)

        # Geometry heads (NOT shared)
        self.ctx_offset = OffsetMLP(feat_dim)
        self.pred_offset = OffsetMLP(feat_dim)
        self.final_offset = OffsetMLP(feat_dim)

    def forward(self, ctx_xyz, ctx_tokens, pred_tokens, edge_index_fn):
        """
        ctx_xyz     : [B, P, 3]
        ctx_tokens  : [B, P, 1024]
        pred_tokens : [B, M, 1024]
        """

        B, P, _ = ctx_xyz.shape
        M = pred_tokens.shape[1]

        # =====================================================
        # Context branch
        # =====================================================
        ctx_feat = self.feat_proj(ctx_tokens)              # [B, P, C]
        xyz_ctx, feat_ctx = self.ctx_upsample(ctx_xyz, ctx_feat)
        # xyz_ctx, feat_ctx : [B, P*r, 3], [B, P*r, C]

        feat_ctx = apply_edgeconv_batched(
            self.edge_conv, xyz_ctx, feat_ctx, edge_index_fn
        )

        delta_ctx = self.ctx_offset(feat_ctx)              # [B, P*r, 3]
        xyz_ctx = xyz_ctx + delta_ctx

        # =====================================================
        # Pred branch (per masked region)
        # =====================================================
        pred_xyz_list = []
        pred_feat_list = []

        anchor = ctx_xyz.mean(dim=1)                        # [B, 3]

        for m in range(M):
            token = pred_tokens[:, m]                       # [B, 1024]
            xyz_p, feat_p = self.pred_upsample(token, anchor)
            # [B, r_pred, 3], [B, r_pred, 1024]

            feat_p = apply_edgeconv_batched(
                self.edge_conv, xyz_p, feat_p, edge_index_fn
            )

            delta_p = self.pred_offset(feat_p)              # [B, r_pred, 3]
            xyz_p = xyz_p + delta_p

            pred_xyz_list.append(xyz_p)
            pred_feat_list.append(feat_p)

        # =====================================================
        # Merge (still batched)
        # =====================================================
        xyz_all = torch.cat([xyz_ctx] + pred_xyz_list, dim=1)
        feat_all = torch.cat([feat_ctx] + pred_feat_list, dim=1)
        # xyz_all : [B, N, 3]
        # feat_all: [B, N, C]

        # =====================================================
        # Final refinement
        # =====================================================
        feat_all = apply_edgeconv_batched(
            self.edge_conv, xyz_all, feat_all, edge_index_fn
        )

        delta_final = self.final_offset(feat_all)
        xyz_all = xyz_all + delta_final

        return xyz_all


@torch.no_grad()
def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- dataset ----
    cfg = ModelNetConfig()
    dataset = ModelNetDataset(cfg)
    sample = dataset[10]

    context = sample["context"].unsqueeze(0).to(device)  # [1, 32, 3]
    mask_centers = torch.stack(
        [t["center"] for t in sample["targets"]]
    ).unsqueeze(0).to(device)  # [1, M, 3]

    # ---- JEPA encoder ----
    jepa = DualEncoder().to(device)
    jepa.eval()

    out = jepa(
        xyz_context=context,
        mask_centers=mask_centers,
        xyz_targets=None,
        mode="infer",
    )

    ctx_xyz = out["ctx_xyz"]                     # [1, 32, 3]
    ctx_tokens = out["ctx_tokens"]               # if available
    pred_tokens = out["pred_tokens"].mean(dim=2) # [1, M, C]

    # ---- Decoder ----
    decoder = JEPAPointDecoder(
        feat_dim=128,
        up_r=12,
        pred_r=3
    ).to(device)
    decoder.eval()

    gen_xyz = decoder(
        ctx_xyz=ctx_xyz,
        ctx_tokens=ctx_tokens,
        pred_tokens=pred_tokens,
        edge_index_fn=edge_index_fn
    )  # [1, N, 3]

    visualize_generated(
        ctx_xyz[0].cpu().numpy(),
        gen_xyz[0].cpu().numpy(),
    )


def edge_index_fn(xyz, k=8):
    # xyz: [N, 3]
    return knn_graph(xyz, k=k, loop=False)


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


if __name__=='__main__':
    run_inference()
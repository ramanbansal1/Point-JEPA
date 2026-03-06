import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, dims, act=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# KNN utilities
# ============================================================
def knn_edges(xyz, k=6):
    d = torch.cdist(xyz, xyz)
    return d.topk(k=k + 1, largest=False).indices[:, :, 1:]


# ============================================================
# Edge subdivision
# ============================================================
def subdivide_edges(
    xyz, tok, knn_idx,
    points_per_edge,
    offset_scale=0.08,
    noise_std=0.01,
):
    B, N, C = tok.shape
    k = knn_idx.shape[-1]

    xyz_i = xyz[:, :, None, :]
    tok_i = tok[:, :, None, :]

    xyz_j = torch.take_along_dim(
        xyz[:, None, :, :],
        knn_idx[:, :, :, None],
        dim=2
    )

    tok_j = torch.take_along_dim(
        tok[:, None, :, :],
        knn_idx[:, :, :, None],
        dim=2
    )

    t = torch.linspace(0, 1, points_per_edge, device=xyz.device)
    t = t[None, None, None, :, None]

    base = xyz_i[:, :, :, None, :] * (1 - t) + xyz_j[:, :, :, None, :] * t

    edge = xyz_j - xyz_i
    edge_len = edge.norm(dim=-1, keepdim=True) + 1e-6
    edge_dir = edge / edge_len

    rand = torch.randn_like(edge_dir)
    normal = torch.cross(edge_dir, rand, dim=-1)
    normal = F.normalize(normal + 1e-8, dim=-1)
    normal = normal[:, :, :, None, :]

    offset = offset_scale * edge_len[..., None] * normal * torch.randn_like(base)
    xyz_edge = base + offset + noise_std * torch.randn_like(base)

    tok_edge = tok_i[:, :, :, None, :] * (1 - t) + tok_j[:, :, :, None, :] * t

    return (
        xyz_edge.reshape(B, -1, 3),
        tok_edge.reshape(B, -1, C),
    )


# ============================================================
# Bounded random sampling
# ============================================================
def bounded_random_sample(xyz, n, margin=0.05):
    xyz_min = xyz.min(dim=1, keepdim=True).values
    xyz_max = xyz.max(dim=1, keepdim=True).values
    size = xyz_max - xyz_min

    xyz_min = xyz_min - margin * size
    xyz_max = xyz_max + margin * size

    u = torch.rand(xyz.shape[0], n, 3, device=xyz.device)
    return xyz_min + u * (xyz_max - xyz_min)


# ============================================================
# Point Generator
# ============================================================
class PointGenerator(nn.Module):
    def __init__(
        self,
        token_dim=1024,
        hidden_dim=256,
        stages=3,
        knn_k=6,
    ):
        super().__init__()

        self.knn_k = knn_k
        self.stages = stages

        self.xyz_proj = nn.Linear(3, hidden_dim)
        self.token_reducer = MLP([token_dim, 512, 256, hidden_dim])
        self.token_norm = nn.LayerNorm(hidden_dim)

        self.deform_mlps = nn.ModuleList([
            MLP([3 + hidden_dim + hidden_dim, 256, 128, 3])
            for _ in range(stages)
        ])
        self.deform_gate = MLP([3 + hidden_dim + hidden_dim, 64, 1])

        self.ctx_weight = nn.Parameter(torch.tensor(1.0))
        self.tgt_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, ctx_xyz, ctx_tokens, pred_tokens, mask_centers):
        ctx = self.token_norm(self.token_reducer(ctx_tokens))
        pred = self.token_norm(self.token_reducer(pred_tokens[:, 0]))

        xyz_feat = self.xyz_proj(ctx_xyz - ctx_xyz.mean(dim=1, keepdim=True))
        pred = pred + xyz_feat

        g_mean = ctx.mean(dim=1)

        # =====================================================
        # GRAPH (33 nodes)
        # =====================================================
        graph_xyz = torch.cat([ctx_xyz, mask_centers[:, :1]], dim=1)
        graph_ctx_tok = torch.cat([ctx, g_mean[:, None]], dim=1)
        graph_pred_tok = torch.cat([pred, g_mean[:, None]], dim=1)

        knn = knn_edges(graph_xyz, self.knn_k)

        # =====================================================
        # CONTEXT (1024 = 512 edge + 512 random)
        # =====================================================
        xyz_e, tok_e = subdivide_edges(
            graph_xyz, graph_ctx_tok, knn, points_per_edge=6
        )
        Ne = 512
        xyz_e, tok_e = xyz_e[:, :Ne], tok_e[:, :Ne]

        xyz_r = bounded_random_sample(graph_xyz, 512)
        d = torch.cdist(xyz_r, graph_xyz)
        idx = d.argmin(dim=-1)
        tok_r = graph_ctx_tok.gather(
            1, idx[..., None].expand(-1, -1, graph_ctx_tok.shape[-1])
        )

        xyz_ctx = torch.cat([xyz_e, xyz_r], dim=1)
        tok_ctx = torch.cat([tok_e, tok_r], dim=1)

        # =====================================================
        # TARGET (6048 = 3024 edge + 3024 random)
        # =====================================================
        xyz_e, tok_e = subdivide_edges(
            graph_xyz, graph_pred_tok, knn, points_per_edge=32
        )
        Ne = 3024
        xyz_e, tok_e = xyz_e[:, :Ne], tok_e[:, :Ne]

        xyz_r = bounded_random_sample(graph_xyz, 3024)
        d = torch.cdist(xyz_r, graph_xyz)
        idx = d.argmin(dim=-1)
        tok_r = graph_pred_tok.gather(
            1, idx[..., None].expand(-1, -1, graph_pred_tok.shape[-1])
        )

        xyz_tgt = torch.cat([xyz_e, xyz_r], dim=1)
        tok_tgt = torch.cat([tok_e, tok_r], dim=1)

        # =====================================================
        # MERGE + DEFORM
        # =====================================================
        xyz = torch.cat([xyz_ctx, xyz_tgt], dim=1)
        tokens = torch.cat([tok_ctx, tok_tgt], dim=1)
        Nc = xyz_ctx.shape[1]

        for _ in range(self.stages):
            g = g_mean[:, None].expand_as(tokens)
            deform_in = torch.cat([xyz, tokens, g], dim=-1)

            delta = self.deform_mlps[0](deform_in)
            gate = torch.sigmoid(self.deform_gate(deform_in))

            xyz[:, :Nc] += self.ctx_weight * gate[:, :Nc] * delta[:, :Nc]
            xyz[:, Nc:] += self.tgt_weight * gate[:, Nc:] * delta[:, Nc:]

        return xyz

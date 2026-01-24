import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, knn_graph

# --------------------------------------------------
# Shared EdgeConv (used by both context and target)
# --------------------------------------------------
import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv


class TwoStageDynamicEdgeConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        k1: int = 16,
        k2: int = 8,
        aggr: str = "max",
    ):
        super().__init__()

        # ---- First Dynamic EdgeConv ----
        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            k=k1,
            aggr=aggr,
        )

        # ---- Second Dynamic EdgeConv ----
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            ),
            k=k2,
            aggr=aggr,
        )

    def forward(self, x):
        """
        x : [N, C_in]
        """
        x = self.conv1(x)   # [N, hidden_channels]
        x = self.conv2(x)   # [N, out_channels]
        return x


# --------------------------------------------------
# Context Deformer (light, stabilizing)
# --------------------------------------------------
class ContextDeformer(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, 3)
        )

    def forward(self, xyz, token):
        offset = 0.05 * self.mlp(token)
        return xyz + offset


# --------------------------------------------------
# Folding-style MLP (target generation)
# --------------------------------------------------
class FoldingMLP(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, xyz, token):
        x = torch.cat([xyz, token], dim=-1)
        return xyz + self.mlp(x)


# --------------------------------------------------
# GNN-based local refinement
# --------------------------------------------------
class GNNRefiner(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.edgeconv = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * token_dim + 3, token_dim),
                nn.ReLU(),
                nn.Linear(token_dim, 3)
            ),
            aggr="mean"
        )

    def forward(self, xyz, token):
        edge_index = knn_graph(xyz, k=16, loop=False)
        x = torch.cat([token, xyz], dim=-1)
        delta = self.edgeconv(x, edge_index)
        return xyz + delta


# --------------------------------------------------
# Generator using CONTEXT TOKENS + PREDICTED XYZ
# --------------------------------------------------
class PointGenerator(nn.Module):
    def __init__(self, token_dim, up_rate=4):
        super().__init__()
        self.up_rate = up_rate

        self.shared_edgeconv = SharedEdgeConv(token_dim)
        self.context_def = ContextDeformer(token_dim)
        self.folding = FoldingMLP(token_dim)
        self.refiner = GNNRefiner(token_dim)

    def forward(self, ctx_xyz, ctx_tokens, pred_xyz):
        """
        ctx_xyz    : [B, P, 3]      (observed context points)
        ctx_tokens : [B, P, C]      (context tokens only)
        pred_xyz   : [B, P, 3]      (predicted coarse xyz from predictor)
        """
        B, P, _ = ctx_xyz.shape
        C = ctx_tokens.size(-1)

        # -------- Context branch --------
        ctx_xyz_f = ctx_xyz.reshape(-1, 3)
        ctx_tok_f = ctx_tokens.reshape(-1, C)

        ctx_latent = self.shared_edgeconv(ctx_xyz_f, ctx_tok_f)
        ctx_xyz_def = self.context_def(ctx_xyz_f, ctx_latent)

        # -------- Target branch --------
        # Upsample predicted xyz
        tgt_xyz = pred_xyz.repeat_interleave(self.up_rate, dim=1)
        noise = torch.randn_like(tgt_xyz) * 0.02
        tgt_xyz = (tgt_xyz + noise).reshape(-1, 3)

        # Token assignment (reuse context tokens)
        tgt_tok = ctx_tokens.repeat_interleave(self.up_rate, dim=1)
        tgt_tok = tgt_tok.reshape(-1, C)

        # Shared EdgeConv
        tgt_latent = self.shared_edgeconv(tgt_xyz, tgt_tok)

        # Folding deformation
        tgt_xyz = self.folding(tgt_xyz, tgt_latent)

        # Local refinement
        tgt_xyz = self.refiner(tgt_xyz, tgt_latent)

        # -------- Join --------
        xyz_final = torch.cat([ctx_xyz_def, tgt_xyz], dim=0)
        return xyz_final

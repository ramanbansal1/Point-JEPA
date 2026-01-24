import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, knn_graph, EdgeConv

class Upsampler(nn.Module):
    def __init__(self, up_rate=4, noise_scale=0.02):
        super().__init__()
        self.up_rate = up_rate
        self.noise_scale = noise_scale

    def forward(self, xyz):
        """
        xyz : [B, P, 3]
        return: [B, P*up_rate, 3]
        """
        xyz_up = xyz.repeat_interleave(self.up_rate, dim=1)
        noise = torch.randn_like(xyz_up) * self.noise_scale
        return xyz_up + noise


class TokenAssigner(nn.Module):
    def __init__(self, up_rate=4):
        super().__init__()
        self.up_rate = up_rate

    def forward(self, ctx_tokens):
        """
        ctx_tokens : [B, P, C]
        return     : [B, P*up_rate, C]
        """
        return ctx_tokens.repeat_interleave(self.up_rate, dim=1)


class SharedEdgeConv(nn.Module):
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
            nn.Linear(token_dim // 4, token_dim // 4),
            nn.ReLU(),
            nn.Linear(token_dim // 4, 3)
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

    def forward(self, xyz, feat):
        return xyz + self.mlp(torch.cat([xyz, feat], dim=-1))


# --------------------------------------------------
# GNN-based local refinement
# --------------------------------------------------
class GNNRefiner(nn.Module):
    def __init__(self, token_dim, k=16):
        super().__init__()
        self.k = k
        self.edgeconv = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * token_dim + 3, token_dim),
                nn.ReLU(),
                nn.Linear(token_dim, 3)
            ),
            aggr="max"
        )

    def forward(self, xyz, feat):
        edge_index = knn_graph(xyz, k=self.k, loop=False)
        x = torch.cat([feat, xyz], dim=-1)
        return xyz + self.edgeconv(x, edge_index)

# --------------------------------------------------
# Generator using CONTEXT TOKENS + PREDICTED XYZ
# --------------------------------------------------
class PointGenerator(nn.Module):
    def __init__(self, token_dim, up_rate=4):
        super().__init__()
        self.upsampler = Upsampler(up_rate)
        self.assigner = TokenAssigner(up_rate)
        self.shared_edgeconv = SharedEdgeConv(token_dim, token_dim // 2, token_dim // 4)
        self.context_def = ContextDeformer(token_dim)
        self.folding = FoldingMLP(token_dim)
        self.refiner = GNNRefiner(token_dim)

    def forward(self, ctx_xyz, ctx_tokens, pred_xyz, pred_token):
        """
        ctx_xyz    : [B, P, 3]
        ctx_tokens : [B, P, C]
        pred_xyz   : [B, P, 3]
        """
        B, P, C = ctx_tokens.shape

        # ---- Context path ----
        ctx_xyz_f = ctx_xyz.view(-1, 3)
        ctx_tok_f = ctx_tokens.reshape(-1, C)
        ctx_feat = self.shared_edgeconv(ctx_tok_f)
        ctx_xyz_out = self.context_def(ctx_xyz_f, ctx_feat)

        # ---- Target path ----
        tgt_xyz = self.upsampler(pred_xyz).view(-1, 3)
        tgt_tok = self.assigner(ctx_tokens).view(-1, C)

        tgt_feat = self.shared_edgeconv(tgt_xyz, tgt_tok)
        tgt_xyz = self.folding(tgt_xyz, tgt_feat)
        tgt_xyz = self.refiner(tgt_xyz, tgt_feat)

        # ---- Join ----
        return torch.cat([ctx_xyz_out, tgt_xyz], dim=0)

import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, EdgeConv, knn_graph

# ==================================================
# 1. Latent-to-XYZ Seed Generator
#    (geometry is decoded from tokens)
# ==================================================
class LatentXYZGenerator(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, 3)
        )

    def forward(self, tokens):
        """
        tokens : [B, P, C]
        return : [B, P, 3]
        """
        return self.mlp(tokens)


# ==================================================
# 2. Upsampler Block
# ==================================================
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


# ==================================================
# 3. Token Assignment Block
# ==================================================
class TokenAssigner(nn.Module):
    def __init__(self, up_rate=4):
        super().__init__()
        self.up_rate = up_rate

    def forward(self, tokens):
        """
        tokens : [B, P, C]
        return : [B, P*up_rate, C]
        """
        return tokens.repeat_interleave(self.up_rate, dim=1)


# ==================================================
# 4. Shared Dynamic EdgeConv Block
#    (feature-space neighborhoods, shared)
# ==================================================
class SharedDynamicEdgeConv(nn.Module):
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

        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            k=k1,
            aggr=aggr,
        )

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
        x : [N, C]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ==================================================
# 5. Context Deformer (light, stabilizing)
# ==================================================
class ContextDeformer(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(),
            nn.Linear(token_dim // 2, 3)
        )

    def forward(self, xyz, feat):
        return xyz + 0.05 * self.mlp(feat)


# ==================================================
# 6. Folding-style MLP (target generation)
# ==================================================
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


# ==================================================
# 7. GNN Refinement Block (XYZ-based, last stage)
# ==================================================
class GNNRefiner(nn.Module):
    def __init__(self, token_dim, k=16):
        super().__init__()
        self.k = k
        self.edgeconv = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * (token_dim + 3), token_dim),
                nn.ReLU(),
                nn.Linear(token_dim, 3)
            ),
            aggr="max"
        )

    def forward(self, xyz, feat):
        edge_index = knn_graph(xyz, k=self.k, loop=False)
        x = torch.cat([feat, xyz], dim=-1)
        return xyz + self.edgeconv(x, edge_index)


# ==================================================
# 8. Generator
#    INPUT SIGNATURE MATCHES:
#    generator(ctx_xyz, ctx_tokens, pred_tokens)
# ==================================================
class PointGenerator(nn.Module):
    def __init__(self, token_dim, up_rate=4):
        super().__init__()
        self.latent_xyz = LatentXYZGenerator(token_dim)
        self.upsampler = Upsampler(up_rate)
        self.assigner = TokenAssigner(up_rate)

        self.shared_dynconv = SharedDynamicEdgeConv(
            token_dim,
            token_dim // 2,
            token_dim // 4,
        )

        self.context_def = ContextDeformer(token_dim // 4)
        self.folding = FoldingMLP(token_dim // 4)
        self.refiner = GNNRefiner(token_dim // 4)

    def forward(self, ctx_xyz, ctx_tokens, pred_tokens, mask_id=0):
        """
        ctx_xyz     : [B, P, 3]
        ctx_tokens  : [B, P, C]          (context tokens)
        pred_tokens : [B, M, P, C]
        mask_id     : which mask to generate
        """
        B, M, P, C = pred_tokens.shape

        # ---- select ONE mask ----
        pred_tok_m = pred_tokens[:, mask_id]          # [B, P, C]

        # ---- generate seed xyz from predicted tokens ----
        seed_xyz = self.latent_xyz(pred_tok_m)        # [B, P, 3]

        # ---- Context path (anchored) ----
        ctx_xyz_f = ctx_xyz.reshape(-1, 3)
        ctx_tok_f = ctx_tokens.reshape(-1, C)

        ctx_feat = self.shared_dynconv(ctx_tok_f)
        ctx_xyz_out = self.context_def(ctx_xyz_f, ctx_feat)

        # ---- Target path ----
        tgt_xyz = self.upsampler(seed_xyz)             # [B, P*up, 3]
        tgt_xyz_f = tgt_xyz.reshape(-1, 3)

        tgt_tok = self.assigner(pred_tok_m)            # [B, P*up, C]
        tgt_tok_f = tgt_tok.reshape(-1, C)

        tgt_feat = self.shared_dynconv(tgt_tok_f)
        tgt_xyz_f = self.folding(tgt_xyz_f, tgt_feat)
        tgt_xyz_f = self.refiner(tgt_xyz_f, tgt_feat)
        print(ctx_xyz_out.shape, tgt_xyz_f.shape)
        # ---- Join ----
        return torch.cat([ctx_xyz_out, tgt_xyz_f], dim=0)

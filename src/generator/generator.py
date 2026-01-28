import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, EdgeConv, knn_graph

# ==================================================
# 1. Latent-to-XYZ Seed Generator
#    (geometry is decoded from tokens, object-frame)
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
        # tokens : [B, P, C]
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
        # xyz : [B, P, 3]
        xyz_up = xyz.repeat_interleave(self.up_rate, dim=1)
        noise = torch.randn_like(xyz_up) * self.noise_scale
        return xyz_up + noise


# ==================================================
# 3. Token Assignment Block (with stabilization noise)
# ==================================================
class TokenAssigner(nn.Module):
    def __init__(self, up_rate=4, noise_scale=0.01):
        super().__init__()
        self.up_rate = up_rate
        self.noise_scale = noise_scale

    def forward(self, tokens):
        # tokens : [B, P, C]
        tok = tokens.repeat_interleave(self.up_rate, dim=1)
        tok = tok + self.noise_scale * torch.randn_like(tok)
        return tok


# ==================================================
# 4. Target Dynamic EdgeConv Block (feature-space)
# ==================================================
class TargetDynamicEdgeConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k1=16, k2=8):
        super().__init__()

        self.conv1 = DynamicEdgeConv(
            nn.Sequential(
                nn.Linear(2 * in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            k=k1,
            aggr="max",
        )

        self.conv2 = DynamicEdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            ),
            k=k2,
            aggr="max",
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class TargetEdgeConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k=16):
        super().__init__()
        self.k = k

        self.edgeconv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            aggr="max",
        )

        self.edgeconv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            ),
            aggr="max",
        )

    def forward(self, xyz, feat):
        # xyz : [N, 3]
        # feat: [N, C]
        edge_index = knn_graph(xyz, k=self.k, loop=False)

        x = self.edgeconv1(feat, edge_index)
        x = self.edgeconv2(x, edge_index)
        return x


# ==================================================
# 5. Context EdgeConv Block (XYZ-based, stable)
# ==================================================
class ContextEdgeConv(nn.Module):
    def __init__(self, token_dim, k=16):
        super().__init__()
        self.k = k
        self.edgeconv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * (token_dim + 3), token_dim),
                nn.ReLU(),
                nn.Linear(token_dim, token_dim)
            ),
            aggr="max"
        )
        self.edgeconv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * (token_dim + 3), token_dim),
                nn.ReLU(),
                nn.Linear(token_dim, token_dim)
            ),
            aggr="max"
        )

    def forward(self, xyz, feat):
        edge_index = knn_graph(xyz, k=self.k, loop=False)
        x = torch.cat([feat, xyz], dim=-1)
        x = self.edgeconv1(x, edge_index)
        x = torch.cat([x, xyz], dim=-1)
        x = self.edgeconv2(x, edge_index)
        return x


# ==================================================
# 6. Context Deformer (surface-only deformation)
# ==================================================
class ContextDeformer(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim + 3, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, 3)
        )

    def forward(self, xyz, feat):
        return xyz + 0.05 * self.mlp(torch.cat([xyz, feat], dim=-1))


# ==================================================
# 7. Folding-style MLP (target generation)
# ==================================================
class FoldingMLP(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(token_dim + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(token_dim + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, grid, feat):
        # grid : [N, 2]
        x = torch.cat([grid, feat], dim=-1)
        xyz = self.mlp1(x)
        x = torch.cat([xyz, feat], dim=-1)
        xyz = xyz + self.mlp2(x)
        return xyz


# ==================================================
# 8. Local GNN Refiner (single, last stage)
# ==================================================
class GNNRefiner(nn.Module):
    def __init__(self, token_dim, k=8):
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
# 9. Generator
# ==================================================
class ContextUpsampler(nn.Module):
    """Geometry-only upsampler for decoded context (no semantics)."""
    def __init__(self, up_rate=4, noise_scale=1e-4):
        super().__init__()
        self.up_rate = up_rate
        self.noise_scale = noise_scale

    def forward(self, xyz):
        # xyz : [B, P, 3]
        xyz_up = xyz.repeat_interleave(self.up_rate, dim=1)
        return xyz_up + self.noise_scale * torch.randn_like(xyz_up)


# ==================================================
# 9. Generator
# ==================================================
class PointGenerator(nn.Module):
    def __init__(
        self,
        token_dim,
        up_rate=4,
        grid_size=4,
        ctx_up_rate=2,
        target_ctx_points=1024,
        target_tgt_points=6048,
    ):
        super().__init__()

        self.target_ctx_points = target_ctx_points
        self.target_tgt_points = target_tgt_points

        # ---------- Context ----------
        self.context_upsampler = ContextUpsampler(ctx_up_rate)
        self.context_edgeconv = ContextEdgeConv(token_dim)
        self.context_def = ContextDeformer(token_dim)

        # ---------- Target ----------
        self.latent_xyz = LatentXYZGenerator(token_dim)
        self.upsampler = Upsampler(up_rate)
        self.assigner = TokenAssigner(up_rate)

        self.target_edgeconv = TargetEdgeConv(
            token_dim,
            token_dim // 2,
            token_dim // 4,
        )


        self.folding = FoldingMLP(token_dim // 4)
        self.refiner = GNNRefiner(token_dim // 4)

        # fixed 2D folding grid
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, grid_size),
                torch.linspace(-1, 1, grid_size),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        self.register_buffer("fold_grid", grid)

        

    def forward(self, ctx_xyz, ctx_tokens, pred_tokens, mask_id=0):
        """
        ctx_xyz     : [B, P_ctx0, 3]
        ctx_tokens  : [B, P_ctx0, C]
        pred_tokens : [B, M, P_pred, C]
        """
        B, M, P_pred, C = pred_tokens.shape
        print('start *')

        # ==================================================
        # Context path (GEOMETRY ONLY)
        # ==================================================
        while ctx_xyz.shape[1] < self.target_ctx_points:
            ctx_xyz = self.context_upsampler(ctx_xyz)
            ctx_tokens = ctx_tokens.repeat_interleave(
                self.context_upsampler.up_rate, dim=1
            )
        ctx_xyz = ctx_xyz[:, : self.target_ctx_points]
        ctx_tokens = ctx_tokens[:, : self.target_ctx_points]
        ctx_xyz_f = ctx_xyz.reshape(-1, 3)
        ctx_tok_f = ctx_tokens.reshape(-1, C)
        ctx_feat = self.context_edgeconv(ctx_xyz_f, ctx_tok_f)
        ctx_xyz_out = self.context_def(ctx_xyz_f, ctx_feat)
        # ==================================================
        # Target path (GENERATION + LOOPED UPSAMPLING)
        # ==================================================
        pred_tok = pred_tokens[:, mask_id]      # [B, P_pred, C]
        
        # seed geometry
        tgt_xyz = self.latent_xyz(pred_tok)     # [B, P_pred, 3]
        tgt_tok = pred_tok
        # iterative upsampling until target resolution
        while tgt_xyz.shape[1] < self.target_tgt_points:
            tgt_xyz = self.upsampler(tgt_xyz)
            tgt_tok = self.assigner(tgt_tok)
        tgt_xyz = tgt_xyz[:, : self.target_tgt_points]
        tgt_tok = tgt_tok[:, : self.target_tgt_points]

        tgt_xyz_f = tgt_xyz.reshape(-1, 3)
        tgt_tok_f = tgt_tok.reshape(-1, C)
        print("finish 1")
        # feature-space neighborhood
        tgt_feat = self.target_edgeconv(tgt_xyz_f, tgt_tok_f)

        # folding
        grid = self.fold_grid.repeat(
            tgt_feat.shape[0] // self.fold_grid.shape[0] + 1, 1
        )[: tgt_feat.shape[0]]

        tgt_xyz_f = self.folding(grid, tgt_feat)
        tgt_xyz_f = self.refiner(tgt_xyz_f, tgt_feat)
        # ==================================================
        # Merge
        # ==================================================
        out = torch.cat([ctx_xyz_out, tgt_xyz_f], dim=0)
        return out

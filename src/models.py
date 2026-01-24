import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from pytorch3d.ops import (
    sample_farthest_points,
    ball_query,
    knn_points,
)
from utils import fourier_features_3d


class PointNetBlock(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        layers = []
        last = in_channels
        for c in mlp_channels:
            layers += [
                nn.Conv2d(last, c, 1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
            last = c
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, K, N]
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]  # max over K neighbors
        return x  # [B, C_out, N]
"""
class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pointnet = PointNetBlock(in_channels + 3, mlp_channels)

    def forward(self, xyz, features):
        ""
        xyz: [B, N, 3]
        features: [B, C, N] or None
        ""
        B, N, _ = xyz.shape

        # FPS
        new_xyz, fps_idx = sample_farthest_points(xyz, K=self.npoint)

        # Ball query
        _, idx, _ = ball_query(
            new_xyz, xyz,
            radius=self.radius,
            K=self.nsample,
            return_nn=False,
        )

        idx[idx < 0] = 0


        grouped_xyz = xyz[:, None].expand(-1, self.npoint, -1, -1)
        grouped_xyz = torch.gather(
            grouped_xyz, 2, idx[..., None].expand(-1, -1, -1, 3)
        )
        grouped_xyz = grouped_xyz - new_xyz[:, :, None, :]
        grouped_xyz = grouped_xyz.permute(0, 3, 2, 1)

        if features is not None:
            grouped_features = features.unsqueeze(2).expand(
                -1, -1, self.npoint, -1
            )
            grouped_features = torch.gather(
                grouped_features,
                3,
                idx[:, None].expand(-1, features.shape[1], -1, -1)
            )
            grouped_features = grouped_features.permute(0, 1, 3, 2)

            grouped = torch.cat([grouped_xyz, grouped_features], dim=1)

        else:
            grouped = grouped_xyz.permute(0, 3, 2, 1)

        new_features = self.pointnet(grouped)
        return new_xyz, new_features
"""
class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pointnet = PointNetBlock(in_channels + 3, mlp_channels)

    def forward(self, xyz, features):
        """
        xyz:      [B, N, 3]
        features: [B, C, N] or None
        """
        B, N, _ = xyz.shape

        # ---- FPS ----
        new_xyz, _ = sample_farthest_points(xyz, K=self.npoint)
        # [B, npoint, 3]

        # ---- Ball query ----
        _, idx, _ = ball_query(
            new_xyz, xyz,
            radius=self.radius,
            K=self.nsample,
            return_nn=False,
        )
        # [B, npoint, nsample]

        idx[idx < 0] = 0

        # ---- Group xyz ----
        grouped_xyz = xyz[:, None, :, :].expand(-1, self.npoint, -1, -1)
        # [B, npoint, N, 3]

        grouped_xyz = torch.gather(
            grouped_xyz,
            2,
            idx[..., None].expand(-1, -1, -1, 3)
        )
        # [B, npoint, nsample, 3]

        grouped_xyz = grouped_xyz - new_xyz[:, :, None, :]
        # [B, npoint, nsample, 3]

        # ---- Reorder to [B, 3, nsample, npoint] ----
        grouped_xyz = grouped_xyz.permute(0, 3, 2, 1)

        # ---- Group features ----
        if features is not None:
            # features: [B, C, N]
            grouped_features = features[:, :, None, :].expand(
                -1, -1, self.npoint, -1
            )
            # [B, C, npoint, N]

            grouped_features = torch.gather(
                grouped_features,
                3,
                idx[:, None].expand(-1, features.shape[1], -1, -1)
            )
            # [B, C, npoint, nsample]

            grouped_features = grouped_features.permute(0, 1, 3, 2)
            # [B, C, nsample, npoint]

            grouped = torch.cat([grouped_xyz, grouped_features], dim=1)
            # [B, 3+C, nsample, npoint]
        else:
            grouped = grouped_xyz
            # [B, 3, nsample, npoint]

        # ---- PointNet ----
        new_features = self.pointnet(grouped)
        # [B, C_out, npoint]

        return new_xyz, new_features


class Encoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()

        self.sa1 = SetAbstraction(
            npoint=512, radius=0.1, nsample=32,
            in_channels=0, mlp_channels=[latent_dim // 16, latent_dim // 16, latent_dim // 8]
        )

        self.sa2 = SetAbstraction(
            npoint=128, radius=0.25, nsample=64,
            in_channels=128, mlp_channels=[latent_dim // 8, latent_dim // 8, latent_dim // 4]
        )

        self.sa3 = SetAbstraction(
            npoint=32, radius=0.5, nsample=128,
            in_channels=256, mlp_channels=[latent_dim // 4, latent_dim // 2, latent_dim]
        )

    def forward(self, xyz):
        """
        xyz: [B, N, 3]
        """
        xyz1, f1 = self.sa1(xyz, None)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)

        return {
            "level0": {"xyz": xyz,  "feat": None},
            "level1": {"xyz": xyz1, "feat": f1},
            "level2": {"xyz": xyz2, "feat": f2},
            "level3": {"xyz": xyz3, "feat": f3},
        }


class LatentPredictor(nn.Module):
    def __init__(self, dim=1024, hidden_dim=2048, pe_dim=24):
        super().__init__()

        self.token_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(dim + pe_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, ctx_tokens, mask_cond):
        """
        ctx_tokens : [B, P, C]
        mask_cond  : [B, M, C+PE]

        returns    : [B, M, P, C]
        """

        B, P, C = ctx_tokens.shape
        M = mask_cond.shape[1]

        # Expand context for each mask
        ctx = ctx_tokens.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, P, C]

        # Token MLP
        h = self.token_mlp(ctx)  # [B, M, P, C]

        # Gating
        gate = self.gate(mask_cond).unsqueeze(2)  # [B, M, 1, C]
        h = h * gate

        return h


class DualEncoder(nn.Module):
    def __init__(self, latent_dim=1024, ema_decay=0.99, pe_bands=4):
        super().__init__()

        self.latent_dim = latent_dim

        self.context_encoder = Encoder(latent_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.ema_decay = ema_decay
        self.pe_bands = pe_bands
        self.pe_dim = 6 * pe_bands

        # Freeze target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        

        self.mask_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.pe_proj = nn.Linear(self.pe_dim, self.pe_dim)

        self.predictor = LatentPredictor(
            dim=latent_dim,
            hidden_dim=2 * latent_dim,
            pe_dim=self.pe_dim,
        )


    def forward(
        self,
        xyz_context,     # [B, Nc, 3]
        mask_centers,    # [B, M, 3]
        xyz_targets,     # List[M][B, P_t, 3]
        mode="train",
    ):
        # ---- Encode context ----
        ctx_out = self.context_encoder(xyz_context)
        ctx_xyz = ctx_out["level3"]["xyz"]               # [B, P, 3]
        ctx_tokens = ctx_out["level3"]["feat"].transpose(1, 2)  # [B, P, C]

        # ---- Mask conditioning ----
        pe = fourier_features_3d(
            mask_centers,
            num_bands=self.pe_dim // 6,
        )                                                # [B, M, PE]

        pe = self.pe_proj(pe)
        mask_tokens = self.mask_token.expand(
            ctx_tokens.size(0), pe.size(1), -1
        )                                                # [B, M, C]

        mask_cond = torch.cat([mask_tokens, pe], dim=-1) # [B, M, C+PE]

        # ---- Predict ----
        pred_tokens = self.predictor(ctx_tokens, mask_cond)  # [B, M, P, C]

        if mode == "infer":
            return {
                "pred_tokens": pred_tokens,
                "ctx_xyz": ctx_xyz,
            }

        # ---- Target encoding (EMA) ----
        tgt_xyz = []
        target_tokens = []

        with torch.no_grad():
            for b, xyz_t in enumerate(xyz_targets):
                sample_tokens = []
                sample_tgt_xyz = []

                for m in xyz_t:
                    m = m.unsqueeze(0)                    # [1, P_t, 3]
                    tgt_out = self.target_encoder(m)

                    sample_tgt_xyz.append(
                        tgt_out["level3"]["xyz"].squeeze(0)
                    )                                     # [P_t', 3]

                    sample_tokens.append(
                        tgt_out["level3"]["feat"]
                        .transpose(1, 2)
                        .squeeze(0)
                    )                                     # [P_t', C]

                tgt_xyz.append(sample_tgt_xyz)
                target_tokens.append(sample_tokens)

        return {
            "pred_tokens": pred_tokens,      # [B, M, P, C]
            "target_tokens": target_tokens,  # [B, M, P_t', C]
            "ctx_xyz": ctx_xyz,              # [B, P, 3]
            "tgt_xyz": tgt_xyz,              # list[M] of [B, P_t', 3]
            "ctx_tokens": ctx_tokens         # [B, P, C]
        }


    @torch.no_grad()
    def update_target_encoder(self):
        for ctx_param, tgt_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            tgt_param.data.mul_(self.ema_decay)
            tgt_param.data.add_((1.0 - self.ema_decay) * ctx_param.data)

def upsample_repeat(xyz, r, noise_std=0.01):
    """
    xyz: [B, P, 3]
    returns: [B, r*P, 3]
    """
    B, P, _ = xyz.shape
    xyz = xyz.unsqueeze(2).repeat(1, 1, r, 1)   # [B, P, r, 3]
    xyz = xyz.reshape(B, P * r, 3)              # [B, rP, 3]
    xyz = xyz + noise_std * torch.randn_like(xyz)
    return xyz


def assign_tokens(xyz, anchor_xyz, anchor_tokens):
    """
    xyz            : [B, Q, 3]
    anchor_xyz     : [B, P, 3]
    anchor_tokens  : [B, P, C]

    returns        : [B, Q, C]
    """
    _, idx, _ = knn_points(xyz, anchor_xyz, K=1)   # [B, Q, 1]
    idx = idx.squeeze(-1)                          # [B, Q]

    B, Q = idx.shape
    C = anchor_tokens.shape[-1]

    tokens = torch.gather(
        anchor_tokens,
        dim=1,
        index=idx.unsqueeze(-1).expand(B, Q, C)
    )
    return tokens



class RecursivePointGenerator(nn.Module):
    def __init__(
        self,
        latent_dim=1024,
        upsample_factors=(6, 4, 4),  # 32 → . → . → 3072
    ):
        super().__init__()

        self.upsample_factors = upsample_factors

        # simple deformation MLP
        self.deformer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 3),
        )

    def forward(self, ctx_xyz, pred_tokens):
        """
        ctx_xyz     : [B, 32, 3]
        pred_tokens : [B, M, 32, C]

        returns     : [B, 3072, 3]
        """

        # ---- collapse mask dimension ----
        tokens = pred_tokens.mean(dim=1)   # [B, 32, C]
        print(tokens.shape)
        xyz = ctx_xyz                      # [B, 32, 3]

        for r in self.upsample_factors:
            # 1. upsample
            xyz = upsample_repeat(xyz, r)  # [B, Q, 3]

            # 2. assign tokens
            point_tokens = assign_tokens(
                xyz,
                ctx_xyz,
                tokens
            )                               # [B, Q, C]

            # 3. deform
            delta = self.deformer(point_tokens)  # [B, Q, 3]
            xyz = xyz + delta

        return xyz
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims, act=nn.ReLU, out_act=None):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        if out_act is not None:
            layers.append(out_act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PointGenerator(nn.Module):
    """
    JEPA token → point cloud generator
    Output: 7072 points
    """

    def __init__(self, token_dim=1024, hidden_dim=256):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ---- Token reducer (MANDATORY)
        self.token_reducer = MLP([
            token_dim, 512, 256, hidden_dim
        ])

        # ---- Token gate
        self.token_gate = MLP([
            hidden_dim * 3, 128, 1
        ])

        # ---- FoldingNet MLPs
        # Input = grid(2) + global(2C') + token(C')
        fold_in = 2 + 2 * hidden_dim + hidden_dim

        self.fold1 = MLP([fold_in, 512, 256, 3])
        self.fold2 = MLP([3 + 2 * hidden_dim, 256, 3])

        # ---- Gate for folding-2
        self.fold_gate = MLP([3 + 2 * hidden_dim, 64, 1])

        # ---- Shared 2D grid (learned)
        self.register_buffer(
            "grid_ctx", self.build_grid(1024)
        )
        self.register_buffer(
            "grid_pred", self.build_grid(6048)
        )

    @staticmethod
    def build_grid(n):
        """Simple square grid"""
        side = int(n ** 0.5) + 1
        lin = torch.linspace(-1, 1, side)
        grid = torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1)
        grid = grid.reshape(-1, 2)[:n]
        return grid
    
    def gated_expand(self, tokens, g_mean, g_max, repeat):
        """
        tokens : [B, 32, C']
        return : [B, repeat*32, C']
        """
        B, P, C = tokens.shape

        g_mean_exp = g_mean[:, None, :].expand(-1, P, -1)
        g_max_exp  = g_max[:, None, :].expand(-1, P, -1)

        gate_in = torch.cat([tokens, g_mean_exp, g_max_exp], dim=-1)
        alpha = torch.sigmoid(self.token_gate(gate_in))  # [B,32,1]

        tokens = alpha * tokens + (1 - alpha) * g_mean_exp
        tokens = tokens.repeat_interleave(repeat, dim=1)
        return tokens

    def fold(self, tokens, grid, g_mean, g_max):
        """
        tokens : [B, N, C']
        grid   : [N, 2]
        """
        B, N, _ = tokens.shape

        grid = grid[None].expand(B, -1, -1)
        g = torch.cat([g_mean, g_max], dim=-1)
        g = g[:, None, :].expand(B, N, -1)

        x1_in = torch.cat([grid, g, tokens], dim=-1)
        x1 = self.fold1(x1_in)

        gate_in = torch.cat([x1, g], dim=-1)
        beta = torch.sigmoid(self.fold_gate(gate_in))

        dx = self.fold2(gate_in)
        x2 = x1 + beta * dx
        return x2
    
    def forward(self, ctx_tokens, pred_tokens):
        """
        ctx_tokens  : [B, 32, 1024]
        pred_tokens : [B, 32, 1024]
        """

        # ---- Reduce tokens
        ctx = self.token_reducer(ctx_tokens)
        pred = self.token_reducer(pred_tokens)

        # ---- Global code from context ONLY
        g_mean = ctx.mean(dim=1)
        g_max  = ctx.max(dim=1).values

        # ---- Context → 1024 points
        ctx_up = self.gated_expand(ctx, g_mean, g_max, repeat=32)
        xyz_ctx = self.fold(ctx_up, self.grid_ctx, g_mean, g_max)

        # ---- Prediction → 6048 points
        pred_up = self.gated_expand(pred[:, 0], g_mean, g_max, repeat=189)
        xyz_pred = self.fold(pred_up, self.grid_pred, g_mean, g_max)

        # ---- Final output
        xyz = torch.cat([xyz_ctx, xyz_pred], dim=1)
        return xyz

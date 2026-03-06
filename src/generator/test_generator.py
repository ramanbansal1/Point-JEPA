import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoder.models import DualEncoder
from generator.gen_data import ModelNetDataset, ModelNetConfig, gen_collate_fn
import logging
logging.getLogger("trimesh").setLevel(logging.ERROR)

def chamfer_loss(x, y):
    d = torch.cdist(x, y)
    return d.min(dim=2)[0].mean() + d.min(dim=1)[0].mean()


class TokenToPoint(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 3),
        )

    def forward(self, tokens, mask_center):
        offsets = self.mlp(tokens)
        return mask_center[:, None, :] + offsets


class PointRefiner(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim + 3, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Linear(token_dim // 2, 3)
        self.prob_head = nn.Linear(token_dim // 2, 1)

    def forward(self, points, tokens):
        x = torch.cat([points, tokens], dim=-1)
        h = self.mlp(x)
        delta = self.offset_head(h)
        prob = torch.sigmoid(self.prob_head(h))
        return delta, prob


class MaskedPointCompletionModel(nn.Module):
    def __init__(self, encoder, token_dim=1024):
        super().__init__()
        self.encoder = encoder
        self.token_to_point = TokenToPoint(token_dim)
        self.refiner = PointRefiner(token_dim)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, ctx_xyz, mask_center):
        with torch.no_grad():
            enc = self.encoder(
                xyz_context=ctx_xyz,
                mask_centers=mask_center[:, None],
                xyz_targets=None,
                mode="infer",
            )
            tokens = enc["pred_tokens"][:, 0]
        coarse_pts = self.token_to_point(tokens, mask_center)
        delta, prob = self.refiner(coarse_pts, tokens)
        refined_pts = coarse_pts + (1.0 - prob) * delta
        return coarse_pts, refined_pts, prob


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ModelNetConfig()
    dataset = ModelNetDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: gen_collate_fn(b, cfg, augment=True),
    )

    encoder = DualEncoder().to(device)
    state = torch.load("checkpoints/jepa_step_9.pt", map_location=device)
    encoder.load_state_dict(state, strict=False)

    model = MaskedPointCompletionModel(encoder).to(device)

    optimizer = torch.optim.Adam(
        list(model.token_to_point.parameters()) +
        list(model.refiner.parameters()),
        lr=1e-3,
    )

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for batch in loader:
            ctx = batch["context_xyz"].to(device)
            tgt = batch["target_xyz"].to(device)
            center = batch["mask_center"].to(device)

            _, refined, prob = model(ctx, center)
            loss_geo = chamfer_loss(refined, tgt)
            loss_prob = prob.mean()
            loss = loss_geo + 0.1 * loss_prob

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"[Epoch {epoch:03d}] loss={total_loss / len(loader):.4f}")


if __name__ == "__main__":
    train()

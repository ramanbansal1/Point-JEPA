import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoder.models import DualEncoder
from generator.gen_data import ModelNetDataset, ModelNetConfig, gen_collate_fn

from tqdm import trange


# ============================================================
# Geometry utilities
# ============================================================

def knn_edges(xyz, k=6):
    d = torch.cdist(xyz, xyz)
    return d.topk(k=k + 1, largest=False).indices[:, :, 1:]


def bounded_random_sample(xyz, n, margin=0.05):
    xyz_min = xyz.min(dim=1, keepdim=True).values
    xyz_max = xyz.max(dim=1, keepdim=True).values
    size = xyz_max - xyz_min

    xyz_min = xyz_min - margin * size
    xyz_max = xyz_max + margin * size

    u = torch.rand(xyz.shape[0], n, 3, device=xyz.device)
    return xyz_min + u * (xyz_max - xyz_min)

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


def initial_upsample(
    ctx_xyz,
    mask_centers,
    ctx_tokens=None,
    mask_tokens=None,
    n_points=7072,
    knn_k=6,
    edge_frac=0.65,
):
    """
    Edge-aware, distribution-aware initialization
    """
    device = ctx_xyz.device
    B = ctx_xyz.shape[0]

    # ---- anchors ----
    anchors_xyz = torch.cat([ctx_xyz, mask_centers], dim=1)  # [B, Na, 3]

    if ctx_tokens is not None and mask_tokens is not None:
        anchors_tok = torch.cat([ctx_tokens, mask_tokens], dim=1)
    else:
        anchors_tok = None

    # ---- knn graph ----
    knn_idx = knn_edges(anchors_xyz, knn_k)  # [B, Na, k]

    # ---- point budget ----
    n_edge = int(edge_frac * n_points)
    n_rand = n_points - n_edge

    Na = anchors_xyz.shape[1]
    k = knn_idx.shape[-1]

    points_per_edge = max(1, n_edge // (Na * k))

    # ---- subdivide edges ----
    xyz_edge, tok_edge = subdivide_edges(
        anchors_xyz,
        anchors_tok if anchors_tok is not None else torch.zeros(B, Na, 1, device=device),
        knn_idx,
        points_per_edge=points_per_edge,
        offset_scale=0.05,
        noise_std=0.01,
    )

    # ---- trim safely ----
    xyz_edge = xyz_edge[:, :n_edge]

    # ---- bounded random ----
    xyz_rand = bounded_random_sample(anchors_xyz, n_rand)

    # ---- final ----
    X0 = torch.cat([xyz_edge, xyz_rand], dim=1)

    return X0


import trimesh
import numpy as np

def visualize_x0_trimesh(ctx_xyz, mask_centers, x0):
    """
    ctx_xyz: [Nc, 3]
    mask_centers: [M, 3]
    x0: [N, 3]
    """

    ctx_xyz = ctx_xyz.detach().cpu().numpy()
    mask_centers = mask_centers.detach().cpu().numpy()
    x0 = x0.detach().cpu().numpy()

    # ---- context (blue) ----
    ctx_pc = trimesh.points.PointCloud(
        ctx_xyz,
        colors=np.tile([0, 0, 255, 255], (ctx_xyz.shape[0], 1))
    )

    # ---- mask center (red) ----
    mask_pc = trimesh.points.PointCloud(
        mask_centers,
        colors=np.tile([255, 0, 0, 255], (mask_centers.shape[0], 1))
    )

    # ---- X0 (green) ----
    x0_pc = trimesh.points.PointCloud(
        x0,
        colors=np.tile([0, 255, 0, 255], (x0.shape[0], 1))
    )

    scene = trimesh.Scene([ctx_pc, mask_pc, x0_pc])
    scene.show()


def chamfer_loss(x, y):
    """
    x: [B, N, 3], y: [B, M, 3]
    """
    d = torch.cdist(x, y)
    return d.min(dim=2)[0].mean() + d.min(dim=1)[0].mean()


def repulsion_loss(x, k=8, eps=1e-6):
    """
    Prevent point collapse
    """
    d = torch.cdist(x, x) + torch.eye(x.shape[1], device=x.device)[None] * 1e6
    knn = d.topk(k, largest=False).values
    return torch.mean(1.0 / (knn + eps))

# ============================================================
# Embedding-space loss
# ============================================================
def latent_moment_loss(a, b):
    """
    a: [B, N, C]
    b: [B, M, C]
    """
    return F.mse_loss(a, b)

def sinkhorn_emd(
    x, y,
    eps=0.05,
    iters=50,
):
    """
    Sinkhorn approximation of EMD
    x: [B, N, 3]
    y: [B, M, 3]
    returns: scalar
    """
    B, N, _ = x.shape
    M = y.shape[1]

    # cost matrix
    C = torch.cdist(x, y)  # [B, N, M]

    # uniform marginals
    mu = torch.full((B, N), 1.0 / N, device=x.device)
    nu = torch.full((B, M), 1.0 / M, device=x.device)

    # kernel
    K = torch.exp(-C / eps)

    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    for _ in range(iters):
        u = mu / (K @ v.unsqueeze(-1)).squeeze(-1)
        v = nu / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)

    # transport plan
    T = u.unsqueeze(-1) * K * v.unsqueeze(1)

    return torch.sum(T * C)



class EnergyLogger:
    def __init__(self):
        self.history = {
            "E_geo": [],
            "E_lat": [],
            "E_rep": [],
            "E_emd": [],
            "E_total": [],
        }

    def log(self, E_geo, E_lat, E_rep, E_emd, E_total):
        self.history["E_geo"].append(E_geo.item())
        self.history["E_lat"].append(E_lat.item())
        self.history["E_rep"].append(E_rep.item())
        self.history["E_emd"].append(E_emd.item())
        self.history["E_total"].append(E_total.item())

    def summary(self): return { k: (v[0], v[-1]) for k, v in self.history.items() }

# ============================================================
# Langevin Generator
# ============================================================
class LangevinGenerator:
    def __init__(
        self,
        encoder: DualEncoder,
        num_points=7072,
        steps=30,
        step_size=3e-3,
        noise_scale=1e-2,
        w_geo=1.0,
        w_lat=10.0,
        w_rep=0.005,
    ):
        self.encoder = encoder
        self.num_points = num_points
        self.steps = steps
        self.step_size = step_size
        self.noise_scale = noise_scale

        self.w_geo = w_geo
        self.w_lat = w_lat
        self.w_rep = w_rep

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def generate(
        self,
        ctx_xyz,
        ctx_tokens,
        tgt_tokens,
        mask_centers,
    ):
        # ---- init ----
        X = initial_upsample(
            ctx_xyz,
            mask_centers,
            ctx_tokens=ctx_tokens,
            mask_tokens=torch.zeros_like(ctx_tokens[:, :1]),
            n_points=self.num_points,
        ).detach()


        

        X.requires_grad_(True)

        anchor_xyz = torch.cat([ctx_xyz, mask_centers], dim=1)
        logger = EnergyLogger()

        E_emd = 0.0
        for _ in trange(self.steps):
            if _ % 5 == 0:
                E_emd = sinkhorn_emd(X, anchor_xyz)

            # ---- encode ----
            emb = self.encoder.encode_xyz_only(X)

            # ---- energies ----
            E_geo = chamfer_loss(X, anchor_xyz)
            E_lat = latent_moment_loss(emb, tgt_tokens)
            E_rep = torch.clamp(repulsion_loss(X), max=50.0)

            rep_w = self.w_rep * (1 - _ / self.steps)


            loss = (
                self.w_geo * E_geo
                + self.w_lat * E_lat
                + rep_w * E_rep
            )

            grad = torch.autograd.grad(loss, X)[0]

            # ---- normalize gradients (CRITICAL) ----
            grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-6)

            X = (
                X
                - self.step_size * grad
                + self.noise_scale * torch.randn_like(X)
            )

            X = X.detach().requires_grad_(True)
            logger.log(E_geo, E_lat, E_rep, E_emd, loss)



        print("Energy summary (start → end):")
        for k, (v0, v1) in logger.summary().items():
            print(f"{k:8s}: {v0:.4f} → {v1:.4f}")


        return X.detach()

# ============================================================
# RUNNER
# ============================================================

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Dataset
    data_cfg = ModelNetConfig()
    dataset = ModelNetDataset(data_cfg)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: gen_collate_fn(b, data_cfg, augment=False),
    )

    # ---- Encoder (FROZEN)
    encoder = DualEncoder().to(device)
    ckpt = torch.load("checkpoints/jepa_step_9.pt", map_location=device)
    encoder.load_state_dict(ckpt, strict=False)

    # ---- Langevin generator
    generator = LangevinGenerator(
        encoder,
        num_points=2048,
        steps=20,
    )

    # ---- Visualization (reuse yours)
    from generator.train_generator import visualize_trimesh

    for batch in loader:
        ctx_xyz = batch["context_xyz"].to(device)
        tgt_xyz = batch["target_xyz"].to(device)
        mask_centers = batch["mask_center"].unsqueeze(1).to(device)

        with torch.no_grad():
            enc = encoder(
                xyz_context=ctx_xyz,
                mask_centers=mask_centers,
                xyz_targets=[[tgt_xyz[0]]],
                mode="train",
            )
            X0 = initial_upsample(
                enc["ctx_xyz"],
                mask_centers,
                n_points=2048,
            )

            visualize_x0_trimesh(
                enc["ctx_xyz"][0],   # correct context
                mask_centers[0],     # [1, 3]
                X0[0],               # X₀
            )


        gen_xyz = generator.generate(
            ctx_xyz=enc["ctx_xyz"],
            ctx_tokens=enc["ctx_tokens"],
            tgt_tokens=enc["pred_tokens"][:, 0],
            mask_centers=mask_centers,
        )

        visualize_trimesh(
            ctx_xyz[0],
            tgt_xyz[0],
            gen_xyz[0],
        )
        break


if __name__ == "__main__":
    run()

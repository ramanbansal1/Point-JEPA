import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import trimesh
from sklearn.decomposition import PCA

from gen_final.data import ModelNetConfig, ModelNetDataset, collate_fn
from encoder.models import DualEncoder



def visualize_trimesh(context_xyz, target_xyz, sampled_xyz):
    ctx = context_xyz[0].cpu().numpy()
    tgt = target_xyz[0].cpu().numpy()
    samp = sampled_xyz[0].cpu().numpy()

    pc_ctx = trimesh.points.PointCloud(ctx, colors=[0, 0, 255, 255])
    pc_tgt = trimesh.points.PointCloud(tgt, colors=[0, 255, 0, 255])
    pc_samp = trimesh.points.PointCloud(samp, colors=[255, 0, 0, 255])

    scene = trimesh.Scene([pc_ctx, pc_tgt, pc_samp])
    scene.show()

def bounded_random_full(context_xyz, mask_center, total_points=7024, margin=0.0):
    """
    context_xyz: [B, Nc, 3]
    mask_center: [B, 3]
    """

    B, Nc, _ = context_xyz.shape

    # include mask center in bounds
    mask_expanded = mask_center[:, None, :]
    all_xyz = torch.cat([context_xyz, mask_expanded], dim=1)

    xyz_min = all_xyz.min(dim=1, keepdim=True).values
    xyz_max = all_xyz.max(dim=1, keepdim=True).values

    size = xyz_max - xyz_min

    # optional margin
    xyz_min = xyz_min - margin * size
    xyz_max = xyz_max + margin * size

    u = torch.rand(B, total_points, 3, device=context_xyz.device)

    sampled = xyz_min + u * (xyz_max - xyz_min)

    return sampled

def sample_random_bounded(
    context_xyz,
    mask_center,
    total_points=7024,
):
    return bounded_random_full(
        context_xyz,
        mask_center,
        total_points=total_points,
        margin=0.0,
    )

def knn_graph(xyz, k):
    """
    xyz: [B, N, 3]
    returns knn indices: [B, N, k_eff]
    """
    B, N, _ = xyz.shape

    if N <= 1:
        return None

    k_eff = min(k, N - 1)

    dist = torch.cdist(xyz, xyz)
    knn_idx = dist.topk(k=k_eff + 1, largest=False).indices[:, :, 1:]

    return knn_idx



def compute_pca_direction(xyz):
    """
    xyz: [B, N, 3]
    returns direction: [B, 3]
    """
    B, N, _ = xyz.shape
    dirs = []

    for b in range(B):
        pts = xyz[b].detach().cpu().numpy()
        pca = PCA(n_components=1)
        pca.fit(pts)
        d = torch.tensor(pca.components_[0], dtype=torch.float32)
        d = d / torch.norm(d)
        dirs.append(d)

    return torch.stack(dirs, dim=0).to(xyz.device)


def slice_mask(xyz, direction, t, delta):
    """
    xyz: [B, N, 3]
    direction: [B, 3]
    """
    proj = (xyz * direction[:, None, :]).sum(dim=-1)
    mask = torch.abs(proj - t) < delta
    return mask


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, knn_idx):
        B, N, C = x.shape
        k = knn_idx.shape[-1]

        neighbors = torch.gather(
            x[:, None, :, :].expand(-1, N, -1, -1),
            2,
            knn_idx[:, :, :, None].expand(-1, -1, -1, C)
        )

        agg = neighbors.mean(dim=2)
        out = self.lin(agg)

        return F.relu(out)


class GraphUNet(nn.Module):
    def __init__(self, in_dim=3, hidden=64):
        super().__init__()

        self.enc1 = GCNLayer(in_dim, hidden)
        self.enc2 = GCNLayer(hidden, hidden)

        self.dec1 = GCNLayer(hidden, hidden)
        self.out = nn.Linear(hidden, 3)

    def forward(self, xyz, knn_idx):
        x = self.enc1(xyz, knn_idx)
        x = self.enc2(x, knn_idx)

        x = self.dec1(x, knn_idx)

        delta = self.out(x)
        return delta

class SlicedGraphGenerator(nn.Module):

    def __init__(self, k=16, num_slices=30, delta=0.08):
        super().__init__()

        self.k = k
        self.num_slices = num_slices
        self.delta = delta

        self.unet = GraphUNet(in_dim=3, hidden=64)

    def forward_train(self, xyz):

        B, N, _ = xyz.shape
        device = xyz.device

        direction = compute_pca_direction(xyz)

        ts = torch.linspace(-1, 1, self.num_slices, device=device)

        for t in ts:

            mask = slice_mask(xyz, direction, t, self.delta)

            if mask.sum() == 0:
                continue

            for b in range(B):

                idx = mask[b]
                if idx.sum() < 5:
                    continue

                slice_pts = xyz[b][idx].unsqueeze(0)

                knn_idx = knn_graph(slice_pts, self.k)

                delta = self.unet(slice_pts, knn_idx)

                xyz[b][idx] = xyz[b][idx] + delta.squeeze(0)

        return xyz

    @torch.no_grad()
    def forward_inference(self, xyz):

        return self.forward_train(xyz)



# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    data_cfg = ModelNetConfig(
        root="../data",
        split="test",
    )

    dataset = ModelNetDataset(data_cfg)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, data_cfg, augment=False),
    )

    batch = next(iter(loader))

    context_xyz = batch["context_xyz"]
    target_xyz = batch["target_xyz"]
    mask_center = batch["mask_center"]

    sampled_xyz = sample_random_bounded(
        context_xyz,
        mask_center,
        total_points=7024,
    )

    visualize_trimesh(context_xyz, target_xyz, sampled_xyz)

    print("Context:", context_xyz.shape)
    print("Target:", target_xyz.shape)
    print("Sampled:", sampled_xyz.shape)

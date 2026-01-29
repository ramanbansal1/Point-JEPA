import torch
import os
from typing import Optional, Union, List, Tuple

def fourier_features_3d(xyz, num_bands=4, scale=1.0):
    """
    Random Fourier Features for 3D coordinates
    
    xyz: [B, N, 3] - 3D coordinates
    num_bands: number of frequency bands
    scale: frequency scaling
    
    Returns: [B, N, num_bands * 6] (sin/cos for each band and dimension)
    """
    if xyz.dim() == 2:
        xyz = xyz.unsqueeze(0)
    
    B, N, _ = xyz.shape
    
    # Frequency bands (geometric progression)
    freq_bands = 2.0 ** torch.linspace(0, num_bands - 1, num_bands, device=xyz.device)
    freq_bands = freq_bands * scale
    
    # Apply frequencies to each coordinate
    xyz_expanded = xyz[:, :, :, None]  # [B, N, 3, 1]
    freqs = freq_bands[None, None, None, :]  # [1, 1, 1, num_bands]
    
    angles = xyz_expanded * freqs  # [B, N, 3, num_bands]
    
    # Apply sin and cos
    sin_features = torch.sin(torch.pi * angles)  # [B, N, 3, num_bands]
    cos_features = torch.cos(torch.pi * angles)  # [B, N, 3, num_bands]
    
    # Concatenate
    features = torch.cat([sin_features, cos_features], dim=-1)  # [B, N, 3, 2*num_bands]
    features = features.reshape(B, N, -1)  # [B, N, 6*num_bands]
    
    return features

def save_checkpoint(model, step, path="checkpoints", type_='jepa'):
    os.makedirs(path, exist_ok=True)
    torch.save(
        model.state_dict(),
        f"{path}/{type_}_step_{step}.pt"
    )


def sample_farthest_points(
    points: torch.Tensor,          # (N, P, D)
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List[int], torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    N, P, D = points.shape
    device = points.device

    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.long, device=device)

    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.long, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.long, device=device)

    max_K = int(K.max())

    idx_out = torch.full((N, max_K), -1, dtype=torch.long, device=device)
    pts_out = torch.zeros((N, max_K, D), dtype=points.dtype, device=device)

    for n in range(N):
        Pn = lengths[n]
        Kn = min(K[n].item(), Pn)

        # distances to nearest selected point
        min_dists = torch.full((Pn,), float("inf"), device=device)

        # initial index
        if random_start_point:
            cur = torch.randint(0, Pn, (1,), device=device).item()
        else:
            cur = 0

        for i in range(Kn):
            idx_out[n, i] = cur
            pts_out[n, i] = points[n, cur]

            diff = points[n, :Pn] - points[n, cur]   # (Pn, D)
            dists = (diff ** 2).sum(dim=-1)           # (Pn,)
            min_dists = torch.minimum(min_dists, dists)

            cur = torch.argmax(min_dists).item()

    return pts_out, idx_out


def ball_query(
    p1: torch.Tensor,     # (N, P1, D)
    p2: torch.Tensor,     # (N, P2, D)
    lengths1: Optional[torch.Tensor] = None,
    lengths2: Optional[torch.Tensor] = None,
    K: int = 32,
    radius: float = 0.2,
    return_nn: bool = True,
):
    N, P1, D = p1.shape
    P2 = p2.shape[1]
    device = p1.device
    r2 = radius ** 2

    if lengths1 is None:
        lengths1 = torch.full((N,), P1, dtype=torch.long, device=device)
    if lengths2 is None:
        lengths2 = torch.full((N,), P2, dtype=torch.long, device=device)

    idx = torch.full((N, P1, K), -1, dtype=torch.long, device=device)
    dists = torch.zeros((N, P1, K), device=device)

    for n in range(N):
        for i in range(lengths1[n]):
            diff = p2[n, :lengths2[n]] - p1[n, i]   # (P2, D)
            dist2 = (diff ** 2).sum(-1)              # (P2,)

            mask = dist2 <= r2
            valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)

            if valid_idx.numel() == 0:
                continue

            sel = valid_idx[:K]
            idx[n, i, :sel.numel()] = sel
            dists[n, i, :sel.numel()] = dist2[sel]

    nn = None
    if return_nn:
        nn = torch.zeros((N, P1, K, D), device=device)
        for n in range(N):
            for i in range(P1):
                for k in range(K):
                    j = idx[n, i, k]
                    if j >= 0:
                        nn[n, i, k] = p2[n, j]

    return dists, idx, nn


def knn_points(
    p1: torch.Tensor,     # (N, P1, D)
    p2: torch.Tensor,     # (N, P2, D)
    lengths1: Optional[torch.Tensor] = None,
    lengths2: Optional[torch.Tensor] = None,
    K: int = 1,
    norm: int = 2,
    return_nn: bool = False,
):
    N, P1, D = p1.shape
    P2 = p2.shape[1]
    device = p1.device

    if lengths1 is None:
        lengths1 = torch.full((N,), P1, dtype=torch.long, device=device)
    if lengths2 is None:
        lengths2 = torch.full((N,), P2, dtype=torch.long, device=device)

    dists_out = torch.zeros((N, P1, K), device=device)
    idx_out = torch.zeros((N, P1, K), dtype=torch.long, device=device)

    for n in range(N):
        for i in range(lengths1[n]):
            diff = p2[n, :lengths2[n]] - p1[n, i]

            if norm == 2:
                dists = (diff ** 2).sum(-1)
            elif norm == 1:
                dists = diff.abs().sum(-1)
            else:
                raise ValueError("Only L1 and L2 norms supported")

            knn_dists, knn_idx = torch.topk(
                dists, k=min(K, lengths2[n]), largest=False
            )

            dists_out[n, i, :knn_dists.numel()] = knn_dists
            idx_out[n, i, :knn_idx.numel()] = knn_idx

    nn = None
    if return_nn:
        nn = torch.zeros((N, P1, K, D), device=device)
        for n in range(N):
            for i in range(P1):
                for k in range(K):
                    nn[n, i, k] = p2[n, idx_out[n, i, k]]

    return dists_out, idx_out, nn

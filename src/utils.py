import torch
import os

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

import numpy as np
import torch
import trimesh
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def normalize_unit_cube(points):
    mins = points.min(0)
    maxs = points.max(0)
    center = (mins + maxs) / 2
    scale = (maxs - mins).max() / 2
    return (points - center) / (scale + 1e-8)


def extract_slice(points, direction, t, delta=0.08):
    proj = points @ direction
    mask = torch.abs(proj - t) < delta
    return points[mask], mask


def create_slice_plane(direction, t, size=2.5):
    """
    Create a square plane mesh for visualization.
    Plane equation: direction · x = t
    """
    normal = direction.detach().cpu().numpy()
    center = normal * float(t)

    plane = trimesh.creation.box(extents=(size, size, 0.01))

    z_axis = np.array([0.0, 0.0, 1.0])
    T = trimesh.geometry.align_vectors(z_axis, normal)
    plane.apply_transform(T)

    plane.apply_translation(center)

    plane.visual.face_colors = [200, 200, 200, 120]
    return plane


# --------------------------------------------------
# Main Algorithm
# --------------------------------------------------

def sliced_diffusion_3d(input_points_np,
                        num_slices=50,
                        iters=100,
                        delta=0.08,
                        lr=1e-3):

    # Normalize
    input_points_np = normalize_unit_cube(input_points_np)

    # PCA direction
    pca = PCA(n_components=1)
    pca.fit(input_points_np)

    direction = torch.tensor(
        pca.components_[0],
        dtype=torch.float32,
        device=device
    )
    direction = direction / torch.norm(direction)

    # Target and Source
    target = torch.tensor(input_points_np, dtype=torch.float32, device=device)
    source = torch.rand_like(target) * 2 - 1

    num_points = source.shape[0]

    ts = torch.linspace(-1, 1, num_slices, device=device)

    # Storage:
    # all_slice_changes[slice][epoch] → (N,3)
    all_slice_changes = []

    for t in tqdm(ts, desc="Slices"):

        slice_src, mask_src = extract_slice(source, direction, t, delta)
        slice_tgt, _ = extract_slice(target, direction, t, delta)

        if slice_src.shape[0] == 0 or slice_tgt.shape[0] == 0:
            all_slice_changes.append([])
            continue

        global_indices = torch.where(mask_src)[0]

        slice_src = slice_src.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([slice_src], lr=lr)

        slice_epoch_changes = []

        for epoch in range(iters):

            optimizer.zero_grad()

            x_old = slice_src.detach().clone()

            # Chamfer-style loss
            x_exp = slice_src.unsqueeze(1)
            y_exp = slice_tgt.unsqueeze(0)
            dist = torch.sum((x_exp - y_exp) ** 2, dim=2)

            loss_xy = torch.mean(torch.min(dist, dim=1).values)
            loss_yx = torch.mean(torch.min(dist, dim=0).values)
            loss = loss_xy + loss_yx

            loss.backward()
            optimizer.step()

            # ------------------------------
            # Tangent Projection + Noise
            # ------------------------------
            with torch.no_grad():

                # Nearest neighbor matching (Chamfer forward direction)
                x_exp = slice_src.unsqueeze(1)
                y_exp = slice_tgt.unsqueeze(0)
                dist = torch.sum((x_exp - y_exp) ** 2, dim=2)

                nn_idx = torch.argmin(dist, dim=1)
                matched_target = slice_tgt[nn_idx]

                # Vector from target to source
                diff = slice_src - matched_target

                norm = torch.norm(diff, dim=1, keepdim=True) + 1e-8
                normal = diff / norm

                # Remove normal component (project to tangent plane)
                normal_component = torch.sum(diff * normal, dim=1, keepdim=True) * normal
                tangential_component = diff - normal_component

                slice_src[:] = matched_target + tangential_component

                # --------- Decaying Noise ---------
                noise_initial = 0.02
                noise_final = 0.001

                noise_scale = noise_initial * (1 - epoch / iters) + \
                            noise_final * (epoch / iters)

                slice_src += noise_scale * torch.randn_like(slice_src)


            # Store global displacement
            with torch.no_grad():
                delta_move = slice_src - x_old

                global_delta = torch.zeros(
                    (num_points, 3),
                    device=device
                )
                global_delta[global_indices] = delta_move

                slice_epoch_changes.append(
                    global_delta.detach().cpu()
                )

        # Write slice back to source
        with torch.no_grad():
            source[mask_src] = slice_src.detach()

        all_slice_changes.append(slice_epoch_changes)

    final_source = source.detach().cpu().numpy()

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    dir_np = direction.detach().cpu().numpy()
    proj_source = final_source @ dir_np
    proj_source = (proj_source - proj_source.min()) / \
                  (proj_source.max() - proj_source.min() + 1e-8)

    cmap = plt.get_cmap("turbo")
    colors_source = (cmap(proj_source) * 255).astype(np.uint8)

    source_cloud = trimesh.points.PointCloud(
        final_source,
        colors=colors_source
    )

    # Add slice planes (visualize every 5th)
    planes = []
    for t in ts[::5]:
        planes.append(create_slice_plane(direction, t))

    scene = trimesh.Scene([source_cloud] + planes)
    scene.show()

    return final_source, all_slice_changes

def plot_epoch_magnitude(all_slice_changes):

    # Determine number of epochs from first non-empty slice
    num_epochs = None
    for slice_changes in all_slice_changes:
        if len(slice_changes) > 0:
            num_epochs = len(slice_changes)
            break

    if num_epochs is None:
        print("No slice updates recorded.")
        return

    mean_mags = []
    max_mags = []

    for epoch in range(num_epochs):

        epoch_total = None

        for slice_changes in all_slice_changes:

            if len(slice_changes) == 0:
                continue

            delta = slice_changes[epoch]  # (N,3)

            if epoch_total is None:
                epoch_total = delta.clone()
            else:
                epoch_total += delta

        if epoch_total is None:
            mean_mags.append(0)
            max_mags.append(0)
            continue

        mag = torch.norm(epoch_total, dim=1)

        mean_mags.append(mag.mean().item())
        max_mags.append(mag.max().item())

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(mean_mags, label="Mean Magnitude")
    plt.plot(max_mags, label="Max Magnitude")
    plt.xlabel("Epoch")
    plt.ylabel("Displacement Magnitude")
    plt.title("Per-Epoch Displacement Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    mesh = trimesh.load(
        "data/bathtub/train/bathtub_0010.off",
        process=False
    )

    pts, face_ids = trimesh.sample.sample_surface_even(
        mesh, 7072
    )

    final_points, all_changes = sliced_diffusion_3d(
        pts,
        num_slices=50,
        iters=20
    )

    print("Number of slices:", len(all_changes))

    plot_epoch_magnitude(all_changes)
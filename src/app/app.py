# app_pointcloud.py
# End-to-end flow (FIXED DEPTH GEOMETRY + AUTO fx ESTIMATION):
# Image → click pixel → SAM2 mask → Depth Anything V2 → camera-space point cloud (.ply)
#
# Key fixes:
# - Camera-space projection (no image-plane cone)
# - Automatic focal length (fx) estimation from depth
# - Depth normalization done ONCE (not per-point)

import gradio as gr
import numpy as np
import torch
from PIL import Image
import cv2
from ultralytics import SAM
import trimesh
import tempfile

# --------------------------------------------------
# Device
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------------------------------------------------
# Load SAM2
# --------------------------------------------------
SAM2_WEIGHTS = "/home/raman/Desktop/Projects/Point-JEPA/app/external/sam2_t.pt"
sam_model = SAM(SAM2_WEIGHTS)

# --------------------------------------------------
# Load Depth Anything V2
# --------------------------------------------------
try:
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    encoder = "vits"
    depth_model = DepthAnythingV2(**model_configs[encoder])
    depth_model.load_state_dict(
        torch.load(f"external/depth_anything_v2_vits.pth", map_location="cpu")
    )
    depth_model = depth_model.to(device).eval()
    depth_available = True

except Exception as e:
    print("Depth Anything V2 not available:", e)
    depth_available = False

# --------------------------------------------------
# SAM2 segmentation using clicked pixel
# --------------------------------------------------
def segment_with_sam2(image_np, point):
    x, y = int(point[0]), int(point[1])

    results = sam_model.predict(
        image_np,
        points=[[x, y]],
        labels=[1],
        verbose=False,
    )

    if len(results) == 0 or results[0].masks is None:
        return None

    mask = results[0].masks.data[0].cpu().numpy()
    return mask

# --------------------------------------------------
# Depth estimation
# --------------------------------------------------
def estimate_depth(image_np):
    if not depth_available:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        return gray.astype(np.float32)

    with torch.no_grad():
        depth = depth_model.infer_image(image_np)

    return depth

# --------------------------------------------------
# Automatic focal length estimation from depth (PLANAR, MASK-AWARE)
# --------------------------------------------------
def estimate_fx_from_depth(depth, mask=None, fx_range=(0.5, 1.5), num_planes=300, patch_radius=2, seed=0):
    """
    Geometry-aware focal length estimation using local planarity.
    This replaces the variance-based heuristic.
    """
    rng = np.random.default_rng(seed)
    H, W = depth.shape
    cx, cy = W / 2, H / 2

    if mask is None:
        mask = depth > 0

    ys, xs = np.where(mask & (depth > 0))
    if len(xs) < 500:
        return 0.9 * max(H, W)

    fx_candidates = np.linspace(
        fx_range[0] * max(H, W),
        fx_range[1] * max(H, W),
        32
    )

    idx = rng.choice(len(xs), size=min(num_planes, len(xs)), replace=False)
    xs, ys = xs[idx], ys[idx]

    def plane_residual(fx):
        fy = fx
        total_err = 0.0
        count = 0

        for x0, y0 in zip(xs, ys):
            pts = []
            for dy in range(-patch_radius, patch_radius + 1):
                for dx in range(-patch_radius, patch_radius + 1):
                    x = x0 + dx
                    y = y0 + dy
                    if (
                        0 <= x < W and
                        0 <= y < H and
                        mask[y, x] and
                        depth[y, x] > 0
                    ):
                        Z = depth[y, x]
                        X = (x - cx) * Z / fx
                        Y = (y - cy) * Z / fy
                        pts.append([X, Y, Z])

            if len(pts) < 6:
                continue

            P = np.asarray(pts)
            centroid = P.mean(axis=0)
            Q = P - centroid

            _, _, vh = np.linalg.svd(Q, full_matrices=False)
            normal = vh[-1]

            d = np.abs(Q @ normal)
            total_err += d.mean()
            count += 1

        return total_err / max(count, 1)

    errors = [plane_residual(fx) for fx in fx_candidates]
    return fx_candidates[np.argmin(errors)]

# --------------------------------------------------
# Create CAMERA-SPACE point cloud (correct geometry)
# --------------------------------------------------
def create_point_cloud(image, depth, mask, stride=2, depth_scale=3.0):
    h, w = depth.shape

    # Normalize depth ONCE
    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-8)
    depth = depth * depth_scale

        # Auto-estimate focal length (planarity-based)
    fx = estimate_fx_from_depth(depth, mask)
    fy = fx
    cx = w / 2.0
    cy = h / 2.0

    pts, cols = [], []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if mask[y, x] < 0.5:
                continue

            Z = depth[y, x]
            if Z <= 0:
                continue

            X = (x - cx) * Z / fx
            Y = -(y - cy) * Z / fy

            pts.append([X, Y, Z])
            cols.append(image[y, x] / 255.0)

    pts = np.asarray(pts)
    cols = np.asarray(cols)

    pc = trimesh.points.PointCloud(pts, colors=cols)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    pc.export(tmp.name)

    return tmp.name

# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def process(image, evt: gr.SelectData):
    if image is None:
        return None, None, None, "No image"

    image_np = np.array(image)
    point = evt.index

    mask = segment_with_sam2(image_np, point)
    if mask is None:
        return None, None, None, "SAM2 failed"

    seg_vis = image_np.copy()
    seg_vis[mask > 0.5] = [0, 255, 0]

    depth = estimate_depth(image_np)
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

    pc_path = create_point_cloud(image_np, depth, mask)

    return seg_vis, depth_vis, pc_path, "✓ 3D point cloud generated (auto fx)"

# --------------------------------------------------
# Gradio UI
# --------------------------------------------------
with gr.Blocks(title="Image → SAM2 → Depth Anything → Point Cloud") as demo:
    gr.Markdown("# Image → SAM2 → Depth Anything V2 → 3D Point Cloud (Auto fx)")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Upload image (click to select object)")
            status = gr.Textbox(label="Status", value="Upload image and click")

        with gr.Column():
            seg_out = gr.Image(label="Segmentation")
            depth_out = gr.Image(label="Depth")

    pc_out = gr.File(label="Point Cloud (.ply)")

    img_in.select(
        fn=process,
        inputs=[img_in],
        outputs=[seg_out, depth_out, pc_out, status],
    )

if __name__ == "__main__":
    demo.launch()

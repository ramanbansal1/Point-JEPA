# app_pointcloud.py
# End-to-end flow (FIXED DEPTH GEOMETRY):
# Image → click pixel → SAM2 mask → Depth Anything V2 → camera-space point cloud (.ply)
#
# Key fix vs previous version:
# - Point cloud is now generated in CAMERA COORDINATES
# - Proper (X, Y, Z) projection using pseudo-intrinsics
# - Depth is scaled, centered, and no longer looks flat

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
        torch.load(f"app/external/depth_anything_v2_vits.pth", map_location="cpu")
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
# Create CAMERA-SPACE point cloud (NOT image-plane)
# --------------------------------------------------
def create_point_cloud(image, depth, mask, stride=2, depth_scale=3.0):
    """
    image: HxWx3 RGB
    depth: HxW relative depth
    mask : HxW binary mask

    Output is in camera coordinates:
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    Z = depth
    """

    h, w = depth.shape

    # Normalize and scale depth (critical)
    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-8)
    depth = depth * depth_scale

    # Pseudo camera intrinsics (reasonable defaults)
    fx = fy = .9 * max(h, w)
    cx = w / 2.0
    cy = h / 2.0

    pts = []
    cols = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if mask[y, x] < 0.5:
                continue

            Z = depth[y, x]
            Z = Z - Z.mean()
            Z = Z / (Z.std() + 1e-6)
            Z = Z * depth_scale

            if Z <= 0:
                continue

            X = (x - cx) * Z / fx
            Y = -(y - cy) * Z / fy  # flip Y for right-handed coord

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
    point = evt.index  # (x, y)

    # 1. SAM2
    mask = segment_with_sam2(image_np, point)
    if mask is None:
        return None, None, None, "SAM2 failed"

    # Visualization
    seg_vis = image_np.copy()
    seg_vis[mask > 0.5] = [0, 255, 0]

    # 2. Depth Anything V2
    depth = estimate_depth(image_np)
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

    # 3. CAMERA-SPACE point cloud
    pc_path = create_point_cloud(
        image_np,
        depth,
        mask,
        stride=2,
        depth_scale=3.0,
    )

    return seg_vis, depth_vis, pc_path, "✓ 3D point cloud generated"

# --------------------------------------------------
# Gradio UI
# --------------------------------------------------
with gr.Blocks(title="Image → SAM2 → Depth Anything → Point Cloud") as demo:
    gr.Markdown("# Image → SAM2 → Depth Anything V2 → 3D Point Cloud")

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

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from mesh_to_3dgs_tree import LODTree
import gsplat
from gsplat.rendering import rasterization


GAUSSIAN_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]
)

def read_3dgs_ply(path: Path) -> np.ndarray:
    data = path.read_bytes()
    marker = b"end_header\n"
    pos = data.find(marker)
    if pos < 0:
        raise ValueError(f"Invalid ply header: {path}")
    start = pos + len(marker)
    payload = data[start:]
    item = GAUSSIAN_PLY_DTYPE.itemsize
    if len(payload) % item != 0:
        raise ValueError(f"Payload size mismatch: {path}")
    return np.frombuffer(payload, dtype=GAUSSIAN_PLY_DTYPE)

def build_global_to_level_local(tree: LODTree) -> Tuple[np.ndarray, np.ndarray]:
    n = tree.node_level.shape[0]
    out_level = np.zeros(n, dtype=np.int32)
    out_local = np.zeros(n, dtype=np.int32)
    # 简单遍历够用；后续你也可以缓存成 npz
    for idx in range(n):
        l, local = tree.get_level_local_index(idx)
        out_level[idx] = l
        out_local[idx] = local
    return out_level, out_local

def compute_viewmat_orbit(center: np.ndarray, yaw: float, pitch: float, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    # world->cam rotation R，并返回 camera position C
    C = center + np.array([
        radius * np.cos(pitch) * np.cos(yaw),
        radius * np.cos(pitch) * np.sin(yaw),
        radius * np.sin(pitch),
    ], dtype=np.float64)

    forward = center - C
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up_world)
    if np.linalg.norm(right) < 1e-9:
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up_world)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, forward)

    R = np.stack([right, up, forward], axis=0)  # world->cam
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = R
    view[:3, 3] = -R @ C
    return view, C

def render_frame_gsplat(
    tree: LODTree,
    per_level: Dict[int, np.ndarray],
    global_level: np.ndarray,
    global_local: np.ndarray,
    selected_global: np.ndarray,
    *,
    cam_center: np.ndarray,
    yaw: float,
    pitch: float,
    radius: float,
    width: int,
    height: int,
    fov_y_deg: float,
    device: str,
    sh_degree: int = 0,
):
    # camera
    viewmat, C = compute_viewmat_orbit(cam_center, yaw, pitch, radius)

    # intrinsics
    fov_y = np.deg2rad(fov_y_deg)
    fy = 0.5 * height / np.tan(0.5 * fov_y)
    fx = fy
    cx = width * 0.5
    cy = height * 0.5
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    if selected_global.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # gather attributes
    levels = global_level[selected_global].astype(np.int32)
    locals_ = global_local[selected_global].astype(np.int32)
    N = selected_global.size

    means = np.zeros((N, 3), dtype=np.float32)
    quats = np.zeros((N, 4), dtype=np.float32)
    scales = np.zeros((N, 3), dtype=np.float32)
    opacities = np.zeros((N,), dtype=np.float32)
    f_dc = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        l = int(levels[i])
        row = int(locals_[i])
        d = per_level[l][row]
        means[i] = [d["x"], d["y"], d["z"]]
        quats[i] = [d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]]
        scales[i] = np.exp([d["scale_0"], d["scale_1"], d["scale_2"]])  # exp(log_scales)
        opacities[i] = float(d["opacity"])
        f_dc[i] = [d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]]

    # SH degree 0: colors shape [N, K=1, 3]
    colors = f_dc[:, None, :]

    means_t = torch.from_numpy(means).to(device)
    quats_t = torch.from_numpy(quats).to(device)
    scales_t = torch.from_numpy(scales).to(device)
    opacities_t = torch.from_numpy(opacities).to(device)
    colors_t = torch.from_numpy(colors).to(device)

    viewmats_t = torch.from_numpy(viewmat[None, ...]).to(device).float()   # [C=1,4,4] (C cameras)
    Ks_t = torch.from_numpy(K[None, ...]).to(device).float()               # [C=1,3,3]

    render_colors, render_alphas, meta = rasterization(
        means_t, quats_t, scales_t, opacities_t, colors_t,
        viewmats=viewmats_t, Ks=Ks_t,
        width=width, height=height,
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB",
    )

    # render_colors: [C=1, H, W, 3] or [H,W,3] depending on gsplat version; normalize robustly
    rc = render_colors
    if rc.ndim == 4:
        rc = rc[0]
    img = torch.clamp(rc, 0.0, 1.0).detach().cpu().numpy()
    return (img * 255.0).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Render dynamic LOD with gsplat (interactive).")
    parser.add_argument("--tree-index", type=Path, required=True)
    parser.add_argument("--lod-dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov-y", type=float, default=50.0)
    parser.add_argument("--max-nodes", type=int, default=8000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    tree = LODTree.load(args.tree_index)
    tree.build_index()

    # preload all levels ply (只选其中一个文件匹配方式，和你生成时一致)
    per_level: Dict[int, np.ndarray] = {}
    for l in range(tree.root_level + 1):
        matches = sorted(args.lod_dir.glob(f"lod_level_{l:02d}_vox*.ply"))
        if not matches:
            raise FileNotFoundError(f"Missing ply for level {l} in {args.lod_dir}")
        per_level[l] = read_3dgs_ply(matches[0])

    global_level, global_local = build_global_to_level_local(tree)

    xyz = tree.node_xyz.astype(np.float64)
    center = xyz.mean(axis=0)
    extent = np.max(np.linalg.norm(xyz - center[None, :], axis=1)) + 1e-6
    radius = float(extent * 2.0)

    yaw, pitch = 0.3, 0.2
    tau = float(args.tau)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("gsplat dynamic LOD (drag orbit | scroll zoom | up/down tau)")
    img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    im = ax.imshow(img)
    ax.axis("off")

    dragging = False
    last = (0.0, 0.0)

    def redraw():
        nonlocal img
        cam_pos = (center + np.array([
            radius * np.cos(pitch) * np.cos(yaw),
            radius * np.cos(pitch) * np.sin(yaw),
            radius * np.sin(pitch),
        ]))
        selected = tree.cut_by_distance(cam_pos, tau=tau, max_nodes=args.max_nodes)
        selected_np = np.asarray(selected, dtype=np.int64)

        img = render_frame_gsplat(
            tree, per_level, global_level, global_local, selected_np,
            cam_center=center, yaw=yaw, pitch=pitch, radius=radius,
            width=args.width, height=args.height,
            fov_y_deg=args.fov_y,
            device=device,
            sh_degree=0,
        )
        im.set_data(img)
        fig.canvas.draw_idle()

    def on_key(e):
        nonlocal tau, radius, yaw, pitch
        if e.key == "up":
            tau += 0.005
            redraw()
        elif e.key == "down":
            tau = max(0.0, tau - 0.005)
            redraw()
        elif e.key in ("r", "R"):
            yaw, pitch = 0.3, 0.2
            radius = float(extent * 2.0)
            redraw()
        elif e.key == "escape":
            plt.close(fig)

    def on_press(e):
        nonlocal dragging, last
        if e.inaxes != ax:
            return
        if e.button == 1:
            dragging = True
            last = (e.xdata if e.xdata is not None else 0.0, e.ydata if e.ydata is not None else 0.0)

    def on_release(e):
        nonlocal dragging
        dragging = False

    def on_move(e):
        nonlocal yaw, pitch, dragging, last
        if not dragging or e.inaxes != ax:
            return
        x = e.xdata if e.xdata is not None else last[0]
        y = e.ydata if e.ydata is not None else last[1]
        dx = float(x - last[0])
        dy = float(y - last[1])
        last = (x, y)
        yaw += dx * 0.005
        pitch -= dy * 0.005
        pitch = float(np.clip(pitch, -1.45, 1.45))
        redraw()

    def on_scroll(e):
        nonlocal radius
        if e.inaxes != ax:
            return
        step = 0.9 if e.button == "up" else 1.1
        radius = float(np.clip(radius * step, extent * 0.2, extent * 20.0))
        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    redraw()
    plt.show()

if __name__ == "__main__":
    main()
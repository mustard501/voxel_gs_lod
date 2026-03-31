#!/usr/bin/env python3
"""
BRDF 多属性可视化：与 lod_dgr_viewer 相同 cut_by_distance + 相机，仅切换传入 CUDA 的「伪球谐 DC」。

每个高斯从 PLY 读 basecolor / normal / metallic / roughness；先将标量或法线映到 [0,1] 三通道，
再按 3DGS 约定做 DC 编码:  sh = (rgb_01 - 0.5) / SH_C0 ，其中 SH_C0 = 0.28209（与 CUDA 一致）。

依赖: torch+cuda, diff_gaussian_rasterization, plyfile, dearpygui

用法:
  python scripts/lod_brdf_attr_viewer.py --lod-dir assets/outputs/2
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SH_C0 = 0.28209


def _src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


if str(_src_dir()) not in sys.path:
    sys.path.insert(0, str(_src_dir()))

from mesh_to_3dgs_tree import LODTree  # noqa: E402


def get_world2view2(
    r: np.ndarray,
    t: np.ndarray,
    translate: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    if translate is None:
        translate = np.zeros(3, dtype=np.float64)
    rt = np.zeros((4, 4), dtype=np.float64)
    rt[:3, :3] = r.T
    rt[:3, 3] = t
    rt[3, 3] = 1.0
    c2w = np.linalg.inv(rt)
    cam_center = c2w[:3, 3].copy()
    cam_center = (cam_center + translate) * scale
    c2w[:3, 3] = cam_center
    rt = np.linalg.inv(c2w)
    return rt.astype(np.float32)


def get_projection_matrix(znear: float, zfar: float, fovx: float, fovy: float) -> np.ndarray:
    import torch

    tan_half_fovy = math.tan((fovy / 2))
    tan_half_fovx = math.tan((fovx / 2))
    top = tan_half_fovy * znear
    bottom = -top
    right = tan_half_fovx * znear
    left = -right
    p = torch.zeros(4, 4, dtype=torch.float32)
    z_sign = 1.0
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = z_sign
    p[2, 2] = z_sign * zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p.numpy()


def look_at_w2c(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    u = up / (np.linalg.norm(up) + 1e-8)
    r = np.cross(u, f)
    r = r / (np.linalg.norm(r) + 1e-8)
    u2 = np.cross(f, r)
    r_cam_to_world = np.stack([r, u2, f], axis=1)
    r_world_to_cam = r_cam_to_world.T
    t = -r_world_to_cam @ eye
    return r_world_to_cam.astype(np.float32), t.astype(np.float32)


def build_camera_tensors(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    znear: float,
    zfar: float,
    fovx: float,
    fovy: float,
    device: "torch.device",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    import torch

    r, t = look_at_w2c(eye, target, up)
    w2v = get_world2view2(r, t)
    wvt = torch.from_numpy(w2v).T.to(device=device, dtype=torch.float32)
    proj = torch.from_numpy(get_projection_matrix(znear, zfar, fovx, fovy)).T.to(
        device=device, dtype=torch.float32
    )
    full_proj = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cam_center = torch.inverse(wvt)[3, :3]
    return wvt, full_proj, cam_center


def rgb01_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N,3) in [0,1] -> (N,3) DC 系数，与 CUDA SH_C0 * dc + 0.5 互逆。"""
    rgb = np.clip(rgb.astype(np.float64), 0.0, 1.0)
    return ((rgb - 0.5) / SH_C0).astype(np.float32)


def load_brdf_ply_multiview(ply_path: Path, device: "torch.device") -> Dict[str, "torch.Tensor"]:
    import torch
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    names = set(v.data.dtype.names or [])

    geom = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    brdf = {"base_r", "base_g", "base_b", "nx", "ny", "nz", "metallic", "roughness"}
    if not geom.issubset(names):
        raise ValueError(f"{ply_path}: 缺少几何/不透明度字段 {geom - names}")
    if not brdf.issubset(names):
        raise ValueError(f"{ply_path}: 缺少 BRDF 字段 {brdf - names}，请使用 BRDF 导出的 PLY。")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    op = np.asarray(v["opacity"], dtype=np.float32).reshape(-1, 1)
    sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

    base = np.stack([v["base_r"], v["base_g"], v["base_b"]], axis=1).astype(np.float64)
    nrm = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float64)
    n_vis = np.clip(nrm * 0.5 + 0.5, 0.0, 1.0)
    met = np.asarray(v["metallic"], dtype=np.float64).reshape(-1, 1)
    rou = np.asarray(v["roughness"], dtype=np.float64).reshape(-1, 1)
    met_rgb = np.clip(np.repeat(met, 3, axis=1), 0.0, 1.0)
    rou_rgb = np.clip(np.repeat(rou, 3, axis=1), 0.0, 1.0)

    sh_b = rgb01_to_sh_dc(base)
    sh_n = rgb01_to_sh_dc(n_vis)
    sh_m = rgb01_to_sh_dc(met_rgb)
    sh_r = rgb01_to_sh_dc(rou_rgb)

    def pack(sh: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(sh[:, None, :]).to(device)

    return {
        "means3D": torch.from_numpy(xyz).to(device),
        "opacities": torch.from_numpy(op).to(device),
        "scales": torch.from_numpy(sc).to(device),
        "rotations": torch.from_numpy(rot).to(device),
        "sh_basecolor": pack(sh_b),
        "sh_normal": pack(sh_n),
        "sh_metallic": pack(sh_m),
        "sh_roughness": pack(sh_r),
        "num": xyz.shape[0],
    }


ATTR_KEYS = {
    "basecolor": "sh_basecolor",
    "normal": "sh_normal",
    "metallic": "sh_metallic",
    "roughness": "sh_roughness",
}


def discover_lod_plies(lod_dir: Path) -> List[Path]:
    pat = re.compile(r"lod_level_(\d+)_", re.I)
    cands = list(lod_dir.glob("lod_level_*_vox*.ply"))
    if not cands:
        cands = list(lod_dir.glob("lod_level_*.ply"))
    if not cands:
        raise FileNotFoundError(f"未找到 lod_level_*.ply: {lod_dir}")

    def sort_key(p: Path) -> Tuple[int, str]:
        m = pat.search(p.name)
        lv = int(m.group(1)) if m else 999
        return (lv, p.name)

    cands.sort(key=sort_key)
    return cands


def gather_brdf_cut(
    tree: LODTree,
    level_tensors: List[Dict[str, "torch.Tensor"]],
    global_indices: List[int],
    device: "torch.device",
    attr: str,
) -> Optional[Dict[str, "torch.Tensor"]]:
    import torch

    if attr not in ATTR_KEYS:
        raise ValueError(f"未知属性: {attr}")
    sh_key = ATTR_KEYS[attr]

    if not global_indices:
        return None

    by_level: Dict[int, List[int]] = defaultdict(list)
    for g in global_indices:
        l, loc = tree.get_level_local_index(int(g))
        by_level[int(l)].append(int(loc))

    pm, po, ps, pr, psh = [], [], [], [], []
    for l in sorted(by_level.keys()):
        idx = torch.tensor(by_level[l], device=device, dtype=torch.long)
        lt = level_tensors[l]
        pm.append(lt["means3D"][idx])
        po.append(lt["opacities"][idx])
        ps.append(lt["scales"][idx])
        pr.append(lt["rotations"][idx])
        psh.append(lt[sh_key][idx])

    return {
        "means3D": torch.cat(pm, dim=0),
        "opacities": torch.cat(po, dim=0),
        "scales": torch.cat(ps, dim=0),
        "rotations": torch.cat(pr, dim=0),
        "shs": torch.cat(psh, dim=0),
        "num": int(sum(t.shape[0] for t in pm)),
    }


def render_frame(gauss: Dict, rasterizer: "GaussianRasterizer") -> "torch.Tensor":
    import torch

    means2d = torch.zeros_like(gauss["means3D"], requires_grad=True, device=gauss["means3D"].device)
    try:
        means2d.retain_grad()
    except Exception:
        pass
    color, _radii = rasterizer(
        means3D=gauss["means3D"],
        means2D=means2d,
        shs=gauss["shs"],
        colors_precomp=None,
        opacities=gauss["opacities"],
        scales=gauss["scales"],
        rotations=gauss["rotations"],
        cov3D_precomp=None,
    )
    return color


def main() -> None:
    import dearpygui.dearpygui as dpg
    import torch

    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    except ImportError as e:
        raise SystemExit(
            "无法导入 diff_gaussian_rasterization，请在 submodules 下: pip install -e .\n" f"{e}"
        ) from e

    parser = argparse.ArgumentParser(description="BRDF 属性通道可视化（同一 cut / 相机）")
    parser.add_argument("--lod-dir", type=Path, required=True)
    parser.add_argument("--w", type=int, default=1280)
    parser.add_argument("--h", type=int, default=720)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--fov-y", type=float, default=60.0)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    lod_dir = args.lod_dir.resolve()
    index_path = lod_dir / "tree_index.npz"
    if not index_path.is_file():
        raise SystemExit(f"未找到 {index_path}")

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA。")

    device = torch.device("cuda")
    tree = LODTree.load(index_path)
    tree.build_index()

    plies = discover_lod_plies(lod_dir)
    level_count = tree.root_level + 1
    if len(plies) != level_count:
        raise SystemExit(f"PLY 层数 {len(plies)} != tree root_level+1={level_count}")

    print("加载 BRDF 各层 PLY（四通道 SH 预计算）…")
    level_tensors: List[Dict] = []
    for p in plies:
        print(f"  {p.name}")
        level_tensors.append(load_brdf_ply_multiview(p, device))

    scene_center = tree.node_xyz.mean(axis=0).astype(np.float64)
    extent = float(np.linalg.norm(tree.node_xyz.max(axis=0) - tree.node_xyz.min(axis=0)) + 1e-8)
    radius = args.radius if args.radius is not None else max(extent * 1.2, 0.5)

    fovy = math.radians(args.fov_y)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * (args.w / args.h))
    znear, zfar = 0.01, 100.0
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    dpg.create_context()
    dpg.create_viewport(title="BRDF attribute LOD viewer", width=args.w + 360, height=args.h + 260)

    tex_w, tex_h = args.w, args.h
    if getattr(dpg, "mvFormat_Float_rgb", None) is None:
        raise SystemExit("dearpygui 需要 mvFormat_Float_rgb")
    _fmt_rgb = dpg.mvFormat_Float_rgb
    tex_buf = np.zeros(tex_w * tex_h * 3, dtype=np.float32)

    with dpg.texture_registry():
        dpg.add_raw_texture(tex_w, tex_h, default_value=tex_buf, format=_fmt_rgb, tag="render_tex")

    r_min = max(extent * 0.05, 0.05)
    r_max = max(extent * 8.0, r_min * 2.0, 5.0)
    attr_items = ["basecolor", "normal", "metallic", "roughness"]

    state = {"tau": float(args.tau), "fps": 0.0, "n_visible": 0}

    with dpg.window(label="Main", tag="win_main", width=args.w + 50, height=args.h + 280):
        dpg.add_image("render_tex", tag="img_view")
        dpg.add_separator()
        dpg.add_text("Display attribute (camera and cut fixed, only change SH DC)")
        dpg.add_combo(
            label="Channel",
            items=attr_items,
            default_value="basecolor",
            tag="combo_attr",
            width=200,
        )
        dpg.add_separator()
        dpg.add_text("Camera (spherical around scene mean)")
        dpg.add_slider_float(
            label="Distance",
            default_value=float(radius),
            min_value=r_min,
            max_value=r_max,
            format="%.3f",
            tag="sl_radius",
        )
        dpg.add_slider_float(
            label="Azimuth theta (deg around Z)",
            default_value=0.0,
            min_value=-180.0,
            max_value=180.0,
            format="%.1f",
            tag="sl_theta_deg",
        )
        dpg.add_slider_float(
            label="Polar phi (deg from +Z)",
            default_value=60.0,
            min_value=5.0,
            max_value=175.0,
            format="%.1f",
            tag="sl_phi_deg",
        )
        dpg.add_separator()
        dpg.add_slider_float(
            label="cut tau",
            default_value=state["tau"],
            min_value=0.001,
            max_value=0.5,
            format="%.4f",
            tag="sl_tau",
        )
        dpg.add_text("", tag="txt_status")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    try:
        dpg.set_primary_window("win_main", True)
    except Exception:
        pass

    print("窗口已打开。combo 切换属性；Q 退出。")

    while dpg.is_dearpygui_running():
        if dpg.is_key_down(dpg.mvKey_Q):
            break
        t0 = time.perf_counter()
        raw_attr = dpg.get_value("combo_attr")
        if isinstance(raw_attr, int):
            attr = attr_items[raw_attr] if 0 <= raw_attr < len(attr_items) else "basecolor"
        else:
            attr = str(raw_attr)
        if attr not in ATTR_KEYS:
            attr = "basecolor"

        cam_r = float(dpg.get_value("sl_radius"))
        theta = math.radians(float(dpg.get_value("sl_theta_deg")))
        phi = math.radians(float(dpg.get_value("sl_phi_deg")))
        state["tau"] = float(dpg.get_value("sl_tau"))

        eye = scene_center + cam_r * np.array(
            [
                math.sin(phi) * math.sin(theta),
                math.sin(phi) * math.cos(theta),
                math.cos(phi),
            ],
            dtype=np.float64,
        )

        cut = tree.cut_by_distance(eye, tau=state["tau"])
        state["n_visible"] = len(cut)
        gauss = gather_brdf_cut(tree, level_tensors, cut, device, attr)

        wvt, full_proj, cam_center = build_camera_tensors(
            eye.astype(np.float32),
            scene_center.astype(np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            znear,
            zfar,
            fovx,
            fovy,
            device,
        )

        rasterizer = GaussianRasterizer(
            raster_settings=GaussianRasterizationSettings(
                image_height=tex_h,
                image_width=tex_w,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg,
                scale_modifier=1.0,
                viewmatrix=wvt,
                projmatrix=full_proj,
                sh_degree=0,
                campos=cam_center,
                prefiltered=False,
            )
        )

        if gauss is None or gauss["num"] == 0:
            tex_buf[:] = 0
        else:
            with torch.no_grad():
                img = render_frame(gauss, rasterizer)
            im = img.detach().clamp(0, 1)
            if im.is_sparse:
                im = im.to_dense()
            while im.dim() > 3:
                im = im.squeeze(0)
            if im.dim() != 3:
                raise RuntimeError(f"bad render shape {tuple(im.shape)}")
            rgb = im.permute(1, 2, 0).contiguous().cpu().numpy()
            tex_buf[:] = rgb.astype(np.float32, copy=False).ravel()

        dpg.set_value("render_tex", tex_buf)

        dt = time.perf_counter() - t0
        state["fps"] = 0.9 * state["fps"] + 0.1 * (1.0 / dt if dt > 1e-6 else 0.0)
        dpg.set_value(
            "txt_status",
            f"status={attr}  |  N={state['n_visible']}  |  r={cam_r:.3f}  tau={state['tau']:.4f}  |  ~{state['fps']:.1f} fps",
        )
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()

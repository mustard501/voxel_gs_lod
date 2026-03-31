#!/usr/bin/env python3
"""
使用 diff_gaussian_rasterization（submodules）实时渲染，并按 LODTree.cut_by_distance
返回的全局节点索引，从各层 PLY 中 gather 高斯子集。

可视化：Dear PyGui

依赖:
  - torch + CUDA, diff_gaussian_rasterization
  - plyfile, dearpygui, numpy

输入目录需包含:
  - tree_index.npz（mesh_to_3dgs_tree 输出）
  - lod_level_00_*.ply … lod_level_L_*.ply（与 npz 层数一致，球谐 f_dc PLY）

用法:
  python scripts/lod_dgr_viewer.py --lod-dir path/to/tree_output

界面滑块：相机距离、水平角 θ、俯仰角 φ（度）、cut 阈值 tau。
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

# -----------------------------------------------------------------------------
# 工程内 LOD 树（cut_by_distance）
# -----------------------------------------------------------------------------
def _src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


if str(_src_dir()) not in sys.path:
    sys.path.insert(0, str(_src_dir()))

from mesh_to_3dgs_tree import LODTree  # noqa: E402


# -----------------------------------------------------------------------------
# 相机矩阵（与 graphdeco-inria/gaussian-splatting 一致）
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# PLY → GPU（球谐）
# -----------------------------------------------------------------------------
def load_sh_ply_gaussians(ply_path: Path, device: "torch.device") -> Dict[str, "torch.Tensor"]:
    import torch
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    names = set(v.data.dtype.names or [])
    need = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    if not need.issubset(names):
        miss = need - names
        raise ValueError(
            f"{ply_path}: 缺少球谐 PLY 字段 {miss}。"
            "请使用 --sh 导出或使用 convert.py 将 BRDF 转为伪 f_dc。"
        )

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    op = np.asarray(v["opacity"], dtype=np.float32).reshape(-1, 1)
    sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    shs = fdc[:, None, :]

    return {
        "means3D": torch.from_numpy(xyz).to(device),
        "opacities": torch.from_numpy(op).to(device),
        "scales": torch.from_numpy(sc).to(device),
        "rotations": torch.from_numpy(rot).to(device),
        "shs": torch.from_numpy(shs).to(device),
        "num": xyz.shape[0],
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


def gather_gaussians_for_cut(
    tree: LODTree,
    level_tensors: List[Dict[str, "torch.Tensor"]],
    global_indices: List[int],
    device: "torch.device",
) -> Optional[Dict[str, "torch.Tensor"]]:
    """按 cut_by_distance 的全局索引，从各层 PLY 张量中索引并拼接。"""
    import torch

    if not global_indices:
        return None

    by_level: Dict[int, List[int]] = defaultdict(list)
    for g in global_indices:
        l, loc = tree.get_level_local_index(int(g))
        by_level[int(l)].append(int(loc))

    parts_means = []
    parts_op = []
    parts_sc = []
    parts_rot = []
    parts_sh = []

    for l in sorted(by_level.keys()):
        idx = torch.tensor(by_level[l], device=device, dtype=torch.long)
        lt = level_tensors[l]
        parts_means.append(lt["means3D"][idx])
        parts_op.append(lt["opacities"][idx])
        parts_sc.append(lt["scales"][idx])
        parts_rot.append(lt["rotations"][idx])
        parts_sh.append(lt["shs"][idx])

    return {
        "means3D": torch.cat(parts_means, dim=0),
        "opacities": torch.cat(parts_op, dim=0),
        "scales": torch.cat(parts_sc, dim=0),
        "rotations": torch.cat(parts_rot, dim=0),
        "shs": torch.cat(parts_sh, dim=0),
        "num": int(sum(t.shape[0] for t in parts_means)),
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
            "无法导入 diff_gaussian_rasterization，请在 CUDA 环境下于 submodules 目录执行: pip install -e .\n"
            f"{e}"
        ) from e

    parser = argparse.ArgumentParser(description="cut_by_distance + Dear PyGui + diff_gaussian_rasterization")
    parser.add_argument("--lod-dir", type=Path, required=True, help="含 tree_index.npz 与各层 lod_level_XX_*.ply")
    parser.add_argument("--w", type=int, default=1280, help="渲染宽")
    parser.add_argument("--h", type=int, default=720, help="渲染高")
    parser.add_argument("--radius", type=float, default=None, help="轨道相机半径（默认由场景估计）")
    parser.add_argument("--fov-y", type=float, default=60.0, help="垂直 FOV（度）")
    parser.add_argument("--tau", type=float, default=0.05, help="cut_by_distance 阈值 tau")
    args = parser.parse_args()

    lod_dir = args.lod_dir.resolve()
    index_path = lod_dir / "tree_index.npz"
    if not index_path.is_file():
        raise SystemExit(f"未找到 {index_path}")

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 与已编译的 diff_gaussian_rasterization。")

    device = torch.device("cuda")
    tree = LODTree.load(index_path)
    tree.build_index()

    plies = discover_lod_plies(lod_dir)
    level_count = tree.root_level + 1
    if len(plies) != level_count:
        raise SystemExit(
            f"PLY 层数 {len(plies)} 与 tree root_level+1={level_count} 不一致。"
            f"请确认目录内有 level 0..{tree.root_level} 的 ply。"
        )

    print("加载各层 PLY → GPU …")
    level_tensors: List[Dict[str, torch.Tensor]] = []
    for p in plies:
        print(f"  {p.name}")
        level_tensors.append(load_sh_ply_gaussians(p, device))

    scene_center = tree.node_xyz.mean(axis=0).astype(np.float64)
    extent = float(np.linalg.norm(tree.node_xyz.max(axis=0) - tree.node_xyz.min(axis=0)) + 1e-8)
    radius = args.radius if args.radius is not None else max(extent * 1.2, 0.5)

    fovy = math.radians(args.fov_y)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * (args.w / args.h))
    znear, zfar = 0.01, 100.0
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    # Dear PyGui
    dpg.create_context()
    dpg.create_viewport(title="LOD cut_by_distance (3DGS CUDA)", width=args.w + 320, height=args.h + 180)

    tex_w, tex_h = args.w, args.h
    # 不同 Dear PyGui 版本格式常量名不一致；优先 float RGB（与当前 PyPI 文档一致）。
    if getattr(dpg, "mvFormat_Float_rgb", None) is not None:
        _fmt_rgb = dpg.mvFormat_Float_rgb
        _tex_dtype = np.float32
    else:
        raise SystemExit(
            "当前 dearpygui 未找到 RGB raw 纹理格式（mvFormat_Float_rgb / UChar / Int）。请升级 dearpygui。"
        )
    tex_buf = np.zeros(tex_w * tex_h * 3, dtype=_tex_dtype)

    with dpg.texture_registry():
        dpg.add_raw_texture(
            tex_w,
            tex_h,
            default_value=tex_buf,
            format=_fmt_rgb,
            tag="render_tex",
        )

    r_min = max(extent * 0.05, 0.05)
    r_max = max(extent * 8.0, r_min * 2.0, 5.0)

    state = {
        "tau": float(args.tau),
        "fps": 0.0,
        "n_visible": 0,
    }

    with dpg.window(label="Main", tag="win_main", width=args.w + 40, height=args.h + 220):
        dpg.add_image("render_tex", tag="img_view")
        dpg.add_separator()
        dpg.add_text("Camera (spherical around scene mean)")
        dpg.add_slider_float(
            label="Camera distance",
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
        dpg.add_text("LOD")
        dpg.add_slider_float(
            label="cut threshold tau",
            default_value=state["tau"],
            min_value=0.001,
            max_value=0.5,
            format="%.4f",
            tag="sl_tau",
        )
        dpg.add_text("status: ", tag="txt_status")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    try:
        dpg.set_primary_window("win_main", True)
    except Exception:
        pass

    print("窗口已打开。拖动滑块调节相机与 tau；按住 Q 退出。")

    while dpg.is_dearpygui_running():
        if dpg.is_key_down(dpg.mvKey_Q):
            break
        t0 = time.perf_counter()
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
        gauss = gather_gaussians_for_cut(tree, level_tensors, cut, device)

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

        raster_settings = GaussianRasterizationSettings(
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
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        if gauss is None or gauss["num"] == 0:
            tex_buf[:] = 0
        else:
            with torch.no_grad():
                img = render_frame(gauss, rasterizer)
            im = img.detach().clamp(0, 1)
            if im.is_sparse:
                im = im.to_dense()
            # diff_gaussian_rasterization 常为 (1, 3, H, W)，去掉 batch 维再转成 HWC
            while im.dim() > 3:
                im = im.squeeze(0)
            if im.dim() != 3:
                raise RuntimeError(f"光栅化输出维度异常: shape={tuple(im.shape)}")
            rgb = im.permute(1, 2, 0).contiguous().cpu().numpy()
            if _tex_dtype == np.float32:
                tex_buf[:] = rgb.astype(np.float32, copy=False).ravel()
            else:
                tex_buf[:] = (rgb * 255.0).astype(np.uint8).ravel()

        dpg.set_value("render_tex", tex_buf)

        dt = time.perf_counter() - t0
        state["fps"] = 0.9 * state["fps"] + 0.1 * (1.0 / dt if dt > 1e-6 else 0.0)
        dpg.set_value(
            "txt_status",
            f"可见高斯: {state['n_visible']}  |  r={cam_r:.3f}  |  tau={state['tau']:.4f}  |  ~{state['fps']:.1f} fps",
        )

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()

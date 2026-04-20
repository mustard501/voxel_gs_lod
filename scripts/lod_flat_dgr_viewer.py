#!/usr/bin/env python3
"""
Render flat hierarchy (tree_index_flat.npz) with BRDF-attribute PLY + Dear PyGui.

- Tree traversal uses child_begin + child_count.
- LOD cut is camera-distance based: refine if (voxel_size(level) / dist(node, cam)) > tau.
- Camera/cut stay fixed while switching view channel:
  basecolor / normal / metallic / roughness.
- Selected attribute is converted to SH-DC by:
  src_data = (src_data - 0.5) / 0.28209
"""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SH_C0 = 0.28209
ATTR_KEYS = {
    "basecolor": "sh_basecolor",
    "normal": "sh_normal",
    "metallic": "sh_metallic",
    "roughness": "sh_roughness",
}


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


@dataclass
class FlatTree:
    parent: np.ndarray
    child_begin: np.ndarray
    child_count: np.ndarray
    level: np.ndarray
    node_xyz: np.ndarray
    base_voxel_size: float
    root_level: int
    root_index: int
    node_to_level_local: np.ndarray

    @staticmethod
    def load(path: Path) -> "FlatTree":
        d = np.load(path, allow_pickle=False)
        parent = d["parent"].astype(np.int32)
        child_begin = d["child_begin"].astype(np.int32)
        child_count = d["child_count"].astype(np.int32)
        level = d["level"].astype(np.int32)
        node_xyz = d["node_xyz"].astype(np.float32)
        base_voxel_size = float(d["base_voxel_size"][0])
        root_level = int(d["root_level"][0])
        roots = np.flatnonzero(parent < 0)
        if roots.size != 1:
            raise ValueError(f"tree_index_flat expects one root, got {roots.size}")
        root_index = int(roots[0])

        node_to_level_local = np.full(parent.shape[0], -1, dtype=np.int32)
        for l in range(root_level + 1):
            ids = np.flatnonzero(level == l)
            node_to_level_local[ids] = np.arange(ids.shape[0], dtype=np.int32)
        if np.any(node_to_level_local < 0):
            raise ValueError("node_to_level_local has invalid entries")

        return FlatTree(
            parent=parent,
            child_begin=child_begin,
            child_count=child_count,
            level=level,
            node_xyz=node_xyz,
            base_voxel_size=base_voxel_size,
            root_level=root_level,
            root_index=root_index,
            node_to_level_local=node_to_level_local,
        )

    def cut_by_distance(self, camera_position: np.ndarray, tau: float = 0.05) -> List[int]:
        cam = np.asarray(camera_position, dtype=np.float64).reshape(3)
        out: List[int] = []
        stack: List[int] = [self.root_index]
        while stack:
            nid = int(stack.pop())
            l = int(self.level[nid])
            d = float(np.linalg.norm(self.node_xyz[nid].astype(np.float64) - cam) + 1e-8)
            voxel_size = float(self.base_voxel_size * (2**l))
            begin = int(self.child_begin[nid])
            count = int(self.child_count[nid])
            refine = (voxel_size / d) > tau and count > 0 and begin >= 0
            if refine:
                for cid in range(begin + count - 1, begin - 1, -1):
                    stack.append(cid)
            else:
                out.append(nid)
        return out


def _to_sh_dc(src_data: np.ndarray) -> np.ndarray:
    return ((src_data - 0.5) / SH_C0).astype(np.float32)


def load_brdf_ply_multiview(ply_path: Path, device: "torch.device") -> Dict[str, "torch.Tensor"]:
    import torch
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    names = set(v.data.dtype.names or [])
    need = {
        "x",
        "y",
        "z",
        "base_r",
        "base_g",
        "base_b",
        "nx",
        "ny",
        "nz",
        "metallic",
        "roughness",
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
        miss = sorted(need - names)
        raise ValueError(f"{ply_path}: missing BRDF fields {miss}")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    op = np.asarray(v["opacity"], dtype=np.float32).reshape(-1, 1)
    sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    base = np.stack([v["base_r"], v["base_g"], v["base_b"]], axis=1).astype(np.float32)
    nrm = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32)
    n_vis = np.clip(0.5 * nrm + 0.5, 0.0, 1.0)
    met = np.asarray(v["metallic"], dtype=np.float32).reshape(-1, 1)
    rou = np.asarray(v["roughness"], dtype=np.float32).reshape(-1, 1)
    met_rgb = np.repeat(met, 3, axis=1)
    rou_rgb = np.repeat(rou, 3, axis=1)

    def pack(sh_dc: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(sh_dc[:, None, :]).to(device)

    return {
        "means3D": torch.from_numpy(xyz).to(device),
        "opacities": torch.from_numpy(op).to(device),
        "scales": torch.from_numpy(sc).to(device),
        "rotations": torch.from_numpy(rot).to(device),
        "sh_basecolor": pack(_to_sh_dc(base)),
        "sh_normal": pack(_to_sh_dc(n_vis)),
        "sh_metallic": pack(_to_sh_dc(met_rgb)),
        "sh_roughness": pack(_to_sh_dc(rou_rgb)),
        "num": xyz.shape[0],
    }


def discover_lod_plies(lod_dir: Path) -> List[Path]:
    pat = re.compile(r"lod_level_(\d+)_", re.I)
    cands = list(lod_dir.glob("lod_level_*_vox*.ply"))
    if not cands:
        cands = list(lod_dir.glob("lod_level_*.ply"))
    if not cands:
        raise FileNotFoundError(f"no lod_level_*.ply in {lod_dir}")

    def key(p: Path) -> Tuple[int, str]:
        m = pat.search(p.name)
        lv = int(m.group(1)) if m else 10**9
        return (lv, p.name)

    cands.sort(key=key)
    return cands


def gather_gaussians_for_cut(
    tree: FlatTree,
    level_tensors: List[Dict[str, "torch.Tensor"]],
    node_indices: List[int],
    device: "torch.device",
    attr: str,
) -> Optional[Dict[str, "torch.Tensor"]]:
    import torch

    if attr not in ATTR_KEYS:
        raise ValueError(f"unknown attr: {attr}")
    sh_key = ATTR_KEYS[attr]

    if not node_indices:
        return None

    by_level: Dict[int, List[int]] = defaultdict(list)
    for nid in node_indices:
        l = int(tree.level[nid])
        loc = int(tree.node_to_level_local[nid])
        by_level[l].append(loc)

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
            "Cannot import diff_gaussian_rasterization. Run `pip install -e .` in submodules first.\n"
            f"{e}"
        ) from e

    parser = argparse.ArgumentParser(description="Flat-tree BRDF attr viewer with child_begin/child_count")
    parser.add_argument("--lod-dir", type=Path, required=True, help="contains tree_index_flat.npz + lod_level_*.ply")
    parser.add_argument("--w", type=int, default=1280)
    parser.add_argument("--h", type=int, default=720)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--fov-y", type=float, default=60.0)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    lod_dir = args.lod_dir.resolve()
    index_path = lod_dir / "tree_index_flat.npz"
    if not index_path.is_file():
        raise SystemExit(f"missing {index_path}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    device = torch.device("cuda")
    tree = FlatTree.load(index_path)
    plies = discover_lod_plies(lod_dir)
    if len(plies) != tree.root_level + 1:
        raise SystemExit(f"ply levels={len(plies)} but root_level+1={tree.root_level + 1}")

    print("Loading BRDF multi-view ply levels to GPU...")
    level_tensors: List[Dict[str, torch.Tensor]] = []
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
    dpg.create_viewport(title="Flat LOD Viewer (3DGS CUDA)", width=args.w + 320, height=args.h + 220)

    tex_w, tex_h = args.w, args.h
    if getattr(dpg, "mvFormat_Float_rgb", None) is None:
        raise SystemExit("dearpygui requires mvFormat_Float_rgb")
    tex_buf = np.zeros(tex_w * tex_h * 3, dtype=np.float32)
    with dpg.texture_registry():
        dpg.add_raw_texture(tex_w, tex_h, default_value=tex_buf, format=dpg.mvFormat_Float_rgb, tag="render_tex")

    r_min = max(extent * 0.05, 0.05)
    r_max = max(extent * 8.0, r_min * 2.0, 5.0)
    state = {"tau": float(args.tau), "fps": 0.0, "n_visible": 0}
    attr_items = ["basecolor", "normal", "metallic", "roughness"]

    with dpg.window(label="Main", tag="win_main", width=args.w + 30, height=args.h + 240):
        dpg.add_image("render_tex", tag="img_view")
        dpg.add_separator()
        dpg.add_text("Flat hierarchy cut: child_begin + child_count")
        dpg.add_combo(
            label="Channel",
            items=attr_items,
            default_value="basecolor",
            tag="combo_attr",
            width=200,
        )
        dpg.add_slider_float(
            label="Distance",
            default_value=float(radius),
            min_value=r_min,
            max_value=r_max,
            format="%.3f",
            tag="sl_radius",
        )
        dpg.add_slider_float(
            label="Azimuth theta (deg)",
            default_value=0.0,
            min_value=-180.0,
            max_value=180.0,
            format="%.1f",
            tag="sl_theta_deg",
        )
        dpg.add_slider_float(
            label="Polar phi (deg from +Z)",
            default_value=90.0,
            min_value=5.0,
            max_value=175.0,
            format="%.1f",
            tag="sl_phi_deg",
        )
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

    print("viewer started; drag sliders, hold Q to quit.")
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
            [math.sin(phi) * math.sin(theta), math.sin(phi) * math.cos(theta), math.cos(phi)],
            dtype=np.float64,
        )

        cut = tree.cut_by_distance(eye, tau=state["tau"])
        state["n_visible"] = len(cut)
        gauss = gather_gaussians_for_cut(tree, level_tensors, cut, device, attr)

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
                raise RuntimeError(f"bad render output shape={tuple(im.shape)}")
            rgb = im.permute(1, 2, 0).contiguous().cpu().numpy()
            tex_buf[:] = rgb.astype(np.float32, copy=False).ravel()

        dpg.set_value("render_tex", tex_buf)
        dt = time.perf_counter() - t0
        state["fps"] = 0.9 * state["fps"] + 0.1 * (1.0 / dt if dt > 1e-6 else 0.0)
        dpg.set_value(
            "txt_status",
            f"attr={attr}  |  N={state['n_visible']}  |  r={cam_r:.3f}  tau={state['tau']:.4f}  |  ~{state['fps']:.1f} fps",
        )
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()


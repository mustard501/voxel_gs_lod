from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

from bake import texture_face_mean_colors
from mesh_io import assert_material_mode_for_path, load_mesh_uniform
from pbr_extract import (
    pbr_face_basecolor_srgb_integrated,
    pbr_face_metallic_roughness_direct,
    pbr_face_shading_normal_integrated,
)


EPS = 1e-8

MaterialMode = Literal["sh", "brdf"]


@dataclass
class FaceRecord:
    center: np.ndarray
    area: float
    normal: np.ndarray
    vertices: np.ndarray
    # 球谐路径
    color: Optional[np.ndarray] = None
    # BRDF 路径（sRGB basecolor 0–1）
    basecolor_srgb: Optional[np.ndarray] = None
    shading_normal: Optional[np.ndarray] = None
    metallic: float = 0.0
    roughness: float = 1.0


@dataclass
class GaussianPrimitive:
    xyz: np.ndarray
    normal: np.ndarray
    covariance: np.ndarray
    opacity: float
    sh_color: Optional[np.ndarray] = None
    basecolor_srgb: Optional[np.ndarray] = None
    metallic: Optional[float] = None
    roughness: Optional[float] = None


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    return load_mesh_uniform(mesh_path)


def _assert_pbr_visual(mesh: trimesh.Trimesh) -> None:
    if not isinstance(mesh.visual.material, PBRMaterial):
        raise ValueError(
            "BRDF export requires a glTF PBR material after load. "
            "Use a GLB/GLTF with PBR, or --sh for OBJ / non-PBR meshes."
        )


def extract_face_records(
    mesh: trimesh.Trimesh,
    *,
    material_mode: MaterialMode,
    texture_samples: int = 256,
    bake_textures: bool = False,
) -> List[FaceRecord]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = vertices[faces]

    centers = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + EPS)

    if material_mode == "sh":
        if bake_textures and texture_samples > 0:
            tex_colors = texture_face_mean_colors(mesh, faces, n_samples=texture_samples)
            face_colors = tex_colors if tex_colors is not None else infer_face_colors(mesh, faces)
        else:
            face_colors = infer_face_colors(mesh, faces)
    else:
        _assert_pbr_visual(mesh)
        if texture_samples <= 0:
            raise ValueError("texture_samples must be positive for BRDF integration.")
        base = pbr_face_basecolor_srgb_integrated(mesh, faces, texture_samples)
        shade = pbr_face_shading_normal_integrated(mesh, faces, normals, texture_samples)
        met, rou = pbr_face_metallic_roughness_direct(mesh, faces)

    records: List[FaceRecord] = []
    for i in range(faces.shape[0]):
        if areas[i] < EPS:
            continue
        if material_mode == "sh":
            records.append(
                FaceRecord(
                    center=centers[i],
                    area=float(areas[i]),
                    normal=normals[i],
                    vertices=tri[i],
                    color=np.asarray(face_colors[i], dtype=np.float64),
                )
            )
        else:
            records.append(
                FaceRecord(
                    center=centers[i],
                    area=float(areas[i]),
                    normal=normals[i],
                    vertices=tri[i],
                    basecolor_srgb=np.asarray(base[i], dtype=np.float64),
                    shading_normal=np.asarray(shade[i], dtype=np.float64),
                    metallic=float(met[i]),
                    roughness=float(rou[i]),
                )
            )
    return records


def infer_face_colors(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    default_color = np.array([0.5, 0.5, 0.5], dtype=np.float64)

    if (
        hasattr(mesh.visual, "vertex_colors")
        and mesh.visual.vertex_colors is not None
        and len(mesh.visual.vertex_colors) == len(mesh.vertices)
    ):
        vc = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0
        return vc[faces].mean(axis=1)

    if (
        hasattr(mesh.visual, "face_colors")
        and mesh.visual.face_colors is not None
        and len(mesh.visual.face_colors) == len(faces)
    ):
        fc = np.asarray(mesh.visual.face_colors[:, :3], dtype=np.float64) / 255.0
        return fc

    return np.repeat(default_color[None, :], repeats=len(faces), axis=0)


def voxel_index(points: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(points / voxel_size).astype(np.int64)


def group_records_by_voxel(
    records: Sequence[FaceRecord], voxel_size: float
) -> Dict[Tuple[int, int, int], List[FaceRecord]]:
    centers = np.stack([r.center for r in records], axis=0)
    indices = voxel_index(centers, voxel_size)
    groups: Dict[Tuple[int, int, int], List[FaceRecord]] = {}
    for idx, rec in zip(indices, records):
        key = (int(idx[0]), int(idx[1]), int(idx[2]))
        groups.setdefault(key, []).append(rec)
    return groups


def weighted_average(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(vectors * weights[:, None], axis=0) / (np.sum(weights) + EPS)


def pca_covariance_from_faces(face_vertices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pts = face_vertices.reshape(-1, 3)
    w = np.repeat(weights, 3)
    mean = weighted_average(pts, w)
    demean = pts - mean[None, :]
    cov = (demean.T @ (demean * w[:, None])) / (np.sum(w) + EPS)

    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-7, None)
    cov_stable = evecs @ np.diag(evals) @ evecs.T
    return cov_stable


def fit_voxel_gaussian(
    face_records: Sequence[FaceRecord], *, material_mode: MaterialMode
) -> GaussianPrimitive:
    areas = np.array([r.area for r in face_records], dtype=np.float64)
    centers = np.stack([r.center for r in face_records], axis=0)
    all_vertices = np.stack([r.vertices for r in face_records], axis=0)

    xyz = weighted_average(centers, areas)
    covariance = pca_covariance_from_faces(all_vertices, areas)
    opacity = 1.0

    if material_mode == "sh":
        normals = np.stack([r.normal for r in face_records], axis=0)
        colors = np.stack([r.color for r in face_records], axis=0)
        normal = weighted_average(normals, areas)
        normal = normal / (np.linalg.norm(normal) + EPS)
        sh_color = weighted_average(colors, areas)
        sh_color = np.clip(sh_color, 0.0, 1.0)
        return GaussianPrimitive(
            xyz=xyz,
            normal=normal,
            covariance=covariance,
            opacity=opacity,
            sh_color=sh_color,
        )

    sn = np.stack([r.shading_normal for r in face_records], axis=0)
    bc = np.stack([r.basecolor_srgb for r in face_records], axis=0)
    mt = np.array([r.metallic for r in face_records], dtype=np.float64)
    rf = np.array([r.roughness for r in face_records], dtype=np.float64)

    normal = weighted_average(sn, areas)
    normal = normal / (np.linalg.norm(normal) + EPS)
    basecolor_srgb = weighted_average(bc, areas)
    basecolor_srgb = np.clip(basecolor_srgb, 0.0, 1.0)
    metallic = float(np.sum(mt * areas) / (np.sum(areas) + EPS))
    roughness = float(np.sum(rf * areas) / (np.sum(areas) + EPS))
    metallic = float(np.clip(metallic, 0.0, 1.0))
    roughness = float(np.clip(roughness, 0.0, 1.0))

    return GaussianPrimitive(
        xyz=xyz,
        normal=normal,
        covariance=covariance,
        opacity=opacity,
        basecolor_srgb=basecolor_srgb,
        metallic=metallic,
        roughness=roughness,
    )


def lod_from_voxel_size(
    records: Sequence[FaceRecord], voxel_size: float, *, material_mode: MaterialMode
) -> List[GaussianPrimitive]:
    groups = group_records_by_voxel(records, voxel_size)
    return [fit_voxel_gaussian(recs, material_mode=material_mode) for recs in groups.values()]


def covariance_to_log_scales_and_quat(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-7, None)
    scales = np.sqrt(evals)
    log_scales = np.log(scales + EPS)
    quat = rotation_matrix_to_quaternion(evecs)
    return log_scales, quat


def rotation_matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    if np.linalg.det(rot) < 0:
        rot[:, 2] *= -1
    m = rot
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + EPS)
    return q


def write_sh_ply(path: Path, gaussians: Sequence[GaussianPrimitive]) -> None:
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(gaussians)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n",
    ]
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    arr = np.zeros(len(gaussians), dtype=dtype)
    for i, g in enumerate(gaussians):
        if g.sh_color is None:
            raise ValueError("SH PLY requires sh_color on each Gaussian.")
        sh = (g.sh_color - 0.5) / 0.28209
        log_scales, quat = covariance_to_log_scales_and_quat(g.covariance)
        arr[i]["x"], arr[i]["y"], arr[i]["z"] = g.xyz.astype(np.float32)
        arr[i]["nx"], arr[i]["ny"], arr[i]["nz"] = g.normal.astype(np.float32)
        arr[i]["f_dc_0"], arr[i]["f_dc_1"], arr[i]["f_dc_2"] = sh.astype(np.float32)
        arr[i]["opacity"] = np.float32(g.opacity)
        arr[i]["scale_0"], arr[i]["scale_1"], arr[i]["scale_2"] = log_scales.astype(np.float32)
        arr[i]["rot_0"], arr[i]["rot_1"], arr[i]["rot_2"], arr[i]["rot_3"] = quat.astype(np.float32)

    with path.open("wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(arr.tobytes())


def write_brdf_ply(path: Path, gaussians: Sequence[GaussianPrimitive]) -> None:
    """BRDF: basecolor sRGB 0–1, world normal, metallic, roughness."""
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(gaussians)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float base_r",
        "property float base_g",
        "property float base_b",
        "property float metallic",
        "property float roughness",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n",
    ]
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("base_r", "<f4"),
            ("base_g", "<f4"),
            ("base_b", "<f4"),
            ("metallic", "<f4"),
            ("roughness", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    arr = np.zeros(len(gaussians), dtype=dtype)
    for i, g in enumerate(gaussians):
        if g.basecolor_srgb is None or g.metallic is None or g.roughness is None:
            raise ValueError("BRDF PLY requires basecolor_srgb, metallic, roughness.")
        log_scales, quat = covariance_to_log_scales_and_quat(g.covariance)
        arr[i]["x"], arr[i]["y"], arr[i]["z"] = g.xyz.astype(np.float32)
        arr[i]["nx"], arr[i]["ny"], arr[i]["nz"] = g.normal.astype(np.float32)
        br, bg, bb = g.basecolor_srgb.astype(np.float32)
        arr[i]["base_r"], arr[i]["base_g"], arr[i]["base_b"] = br, bg, bb
        arr[i]["metallic"] = np.float32(g.metallic)
        arr[i]["roughness"] = np.float32(g.roughness)
        arr[i]["opacity"] = np.float32(g.opacity)
        arr[i]["scale_0"], arr[i]["scale_1"], arr[i]["scale_2"] = log_scales.astype(np.float32)
        arr[i]["rot_0"], arr[i]["rot_1"], arr[i]["rot_2"], arr[i]["rot_3"] = quat.astype(np.float32)

    with path.open("wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(arr.tobytes())


def write_gaussian_ply(path: Path, gaussians: Sequence[GaussianPrimitive], *, material_mode: MaterialMode) -> None:
    if material_mode == "sh":
        write_sh_ply(path, gaussians)
    else:
        write_brdf_ply(path, gaussians)


def export_lods(
    mesh_path: Path,
    output_dir: Path,
    voxel_sizes: Iterable[float],
    *,
    use_sh: bool,
    bake_textures: bool = False,
    texture_samples: int = 256,
) -> None:
    assert_material_mode_for_path(mesh_path, use_sh=use_sh)
    material_mode: MaterialMode = "sh" if use_sh else "brdf"
    mesh = load_mesh_uniform(mesh_path)
    records = extract_face_records(
        mesh,
        material_mode=material_mode,
        texture_samples=texture_samples,
        bake_textures=bake_textures and use_sh,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, voxel_size in enumerate(voxel_sizes):
        gaussians = lod_from_voxel_size(records, voxel_size, material_mode=material_mode)
        out_file = output_dir / f"lod_{i:02d}_vox{voxel_size:.4f}.ply"
        write_gaussian_ply(out_file, gaussians, material_mode=material_mode)
        print(f"[LOD {i}] voxel={voxel_size:.6f}, gaussians={len(gaussians)} -> {out_file}")


def parse_voxel_sizes(text: str) -> List[float]:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("voxel sizes cannot be empty")
    if any(v <= 0 for v in vals):
        raise ValueError("voxel sizes must be positive")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample mesh into multi-LOD Gaussian .ply (default BRDF for GLB; --sh for spherical harmonics)."
    )
    parser.add_argument("--mesh", type=Path, required=True, help="Input mesh (OBJ with --sh, or GLB/GLTF for BRDF).")
    parser.add_argument(
        "--voxel-sizes",
        type=str,
        required=True,
        help="Comma separated voxel sizes, e.g. '0.01,0.02,0.04'.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--sh",
        action="store_true",
        help="Spherical harmonics (f_dc_*) export; required for OBJ. Default is BRDF (GLB/GLTF).",
    )
    parser.add_argument(
        "--bake-textures",
        action="store_true",
        help="(SH only) Per-face texture average for albedo proxy.",
    )
    parser.add_argument(
        "--texture-samples",
        type=int,
        default=256,
        help="Samples per face for BRDF basecolor/normal integration, or SH texture averaging (default 256).",
    )
    args = parser.parse_args()

    voxel_sizes = parse_voxel_sizes(args.voxel_sizes)
    export_lods(
        args.mesh,
        args.out_dir,
        voxel_sizes,
        use_sh=args.sh,
        bake_textures=args.bake_textures,
        texture_samples=args.texture_samples,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

from bake import texture_face_mean_colors


EPS = 1e-8


@dataclass
class FaceRecord:
    center: np.ndarray
    area: float
    normal: np.ndarray
    color: np.ndarray
    vertices: np.ndarray


@dataclass
class GaussianPrimitive:
    xyz: np.ndarray
    normal: np.ndarray
    color: np.ndarray
    covariance: np.ndarray
    opacity: float

# 返回mesh
def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Input is not a triangle mesh: {mesh_path}")
    if mesh.faces.shape[0] == 0:
        raise ValueError("Mesh has no faces.")
    if mesh.vertices.shape[0] == 0:
        raise ValueError("Mesh has no vertices.")
    return mesh


# 返回面片记录
def extract_face_records(
    mesh: trimesh.Trimesh, *, texture_samples: Optional[int] = None
) -> List[FaceRecord]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = vertices[faces]  # (F, 3, 3)

    centers = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + EPS)

    if texture_samples is not None and texture_samples > 0:
        tex_colors = texture_face_mean_colors(mesh, faces, n_samples=texture_samples)
        face_colors = tex_colors if tex_colors is not None else infer_face_colors(mesh, faces)
    else:
        face_colors = infer_face_colors(mesh, faces)

    records: List[FaceRecord] = []
    for i in range(faces.shape[0]):
        if areas[i] < EPS:
            continue
        records.append(
            FaceRecord(
                center=centers[i],
                area=float(areas[i]),
                normal=normals[i],
                color=face_colors[i],
                vertices=tri[i],
            )
        )
    return records


# 返回面片颜色
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


# 返回体素索引
def voxel_index(points: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(points / voxel_size).astype(np.int64)


# 返回体素内面片信息（字典：键为体素索引，值为面片记录列表）
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


# 返回加权平均
def weighted_average(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(vectors * weights[:, None], axis=0) / (np.sum(weights) + EPS)


# 返回PCA协方差
def pca_covariance_from_faces(face_vertices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pts = face_vertices.reshape(-1, 3)  # (N*3, 3)
    w = np.repeat(weights, 3)
    mean = weighted_average(pts, w)
    demean = pts - mean[None, :]
    cov = (demean.T @ (demean * w[:, None])) / (np.sum(w) + EPS)

    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-7, None)
    cov_stable = evecs @ np.diag(evals) @ evecs.T
    return cov_stable


# 返回体素高斯
def fit_voxel_gaussian(face_records: Sequence[FaceRecord]) -> GaussianPrimitive:
    areas = np.array([r.area for r in face_records], dtype=np.float64)
    centers = np.stack([r.center for r in face_records], axis=0)
    normals = np.stack([r.normal for r in face_records], axis=0)
    colors = np.stack([r.color for r in face_records], axis=0)
    all_vertices = np.stack([r.vertices for r in face_records], axis=0)

    xyz = weighted_average(centers, areas)
    normal = weighted_average(normals, areas)
    normal = normal / (np.linalg.norm(normal) + EPS)

    color = weighted_average(colors, areas)
    color = np.clip(color, 0.0, 1.0)

    covariance = pca_covariance_from_faces(all_vertices, areas)
    opacity = 1.0

    return GaussianPrimitive(
        xyz=xyz, normal=normal, color=color, covariance=covariance, opacity=opacity
    )


# 返回特定大小体素的高斯
def lod_from_voxel_size(
    records: Sequence[FaceRecord], voxel_size: float
) -> List[GaussianPrimitive]:
    groups = group_records_by_voxel(records, voxel_size)
    return [fit_voxel_gaussian(recs) for recs in groups.values()]


# 返回对数尺寸和四元数
def covariance_to_log_scales_and_quat(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-7, None)
    scales = np.sqrt(evals)
    log_scales = np.log(scales + EPS)
    quat = rotation_matrix_to_quaternion(evecs)
    return log_scales, quat


# 从旋转矩阵生成四元数
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


# 写入3DGS-style PLY文件
def write_3dgs_ply(path: Path, gaussians: Sequence[GaussianPrimitive]) -> None:
    # Minimal 3DGS-style attribute set.
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
        g.color = (g.color - 0.5)/0.28209
        log_scales, quat = covariance_to_log_scales_and_quat(g.covariance)
        arr[i]["x"], arr[i]["y"], arr[i]["z"] = g.xyz.astype(np.float32)
        arr[i]["nx"], arr[i]["ny"], arr[i]["nz"] = g.normal.astype(np.float32)
        arr[i]["f_dc_0"], arr[i]["f_dc_1"], arr[i]["f_dc_2"] = g.color.astype(np.float32)
        arr[i]["opacity"] = np.float32(g.opacity)
        arr[i]["scale_0"], arr[i]["scale_1"], arr[i]["scale_2"] = log_scales.astype(np.float32)
        arr[i]["rot_0"], arr[i]["rot_1"], arr[i]["rot_2"], arr[i]["rot_3"] = quat.astype(
            np.float32
        )

    with path.open("wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(arr.tobytes())


# 导出多尺度体素高斯
def export_lods(
    mesh_path: Path,
    output_dir: Path,
    voxel_sizes: Iterable[float],
    bake_textures: bool,
    texture_samples: int = 256,
) -> None:
    mesh = load_mesh(mesh_path)
    ts = texture_samples if bake_textures else None
    records = extract_face_records(mesh, texture_samples=ts)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, voxel_size in enumerate(voxel_sizes):
        gaussians = lod_from_voxel_size(records, voxel_size)
        out_file = output_dir / f"lod_{i:02d}_vox{voxel_size:.4f}.ply"
        write_3dgs_ply(out_file, gaussians)
        print(f"[LOD {i}] voxel={voxel_size:.6f}, gaussians={len(gaussians)} -> {out_file}")


# 从命令行参数解析体素尺寸列表
def parse_voxel_sizes(text: str) -> List[float]:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("voxel sizes cannot be empty")
    if any(v <= 0 for v in vals):
        raise ValueError("voxel sizes must be positive")
    return vals


# 主函数
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample mesh into multi-LOD 3D Gaussian Splatting (.ply) models."
    )
    parser.add_argument("--mesh", type=Path, required=True, help="Input triangle mesh file.")
    parser.add_argument(
        "--voxel-sizes",
        type=str,
        required=True,
        help="Comma separated voxel sizes, e.g. '0.01,0.02,0.04'.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--bake-textures",
        action="store_true",
        help="Use UV texture: per-face average via bilinear sampling (no vertex bake).",
    )
    parser.add_argument(
        "--texture-samples",
        type=int,
        default=256,
        help="Samples per face for texture averaging when --bake-textures (default 256).",
    )
    args = parser.parse_args()

    voxel_sizes = parse_voxel_sizes(args.voxel_sizes)
    export_lods(
        args.mesh,
        args.out_dir,
        voxel_sizes,
        bake_textures=args.bake_textures,
        texture_samples=args.texture_samples,
    )


if __name__ == "__main__":
    main()

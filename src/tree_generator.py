from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, cast

import numpy as np

import mesh_to_3dgs_lod as m
from mesh_io import assert_material_mode_for_path
from morton import morton3d_child_key, morton3d_encode, morton3d_parent_key


EPS = 1e-8


@dataclass
class _NodeBuild:
    parent: int
    level: int
    morton_key: int
    records: List[m.FaceRecord]
    child_begin: int
    child_count: int


def _finest_voxel_morton(
    records: List[m.FaceRecord],
    base_voxel_size: float,
    bits_per_axis: int,
    origin_ijk: np.ndarray,
) -> Dict[int, List[m.FaceRecord]]:
    """Finest cells: Morton key -> face records (relative indices to origin_ijk)."""
    centers = np.stack([r.center for r in records], axis=0)
    ijk = m.voxel_index(centers, base_voxel_size) - origin_ijk.astype(np.int64)
    groups: Dict[int, List[m.FaceRecord]] = defaultdict(list)
    for t in range(centers.shape[0]):
        i, j, k = int(ijk[t, 0]), int(ijk[t, 1]), int(ijk[t, 2])
        mk = int(morton3d_encode(i, j, k, bits_per_axis=bits_per_axis))
        groups[mk].append(records[t])
    return dict(groups)


def _build_level_maps(
    finest: Dict[int, List[m.FaceRecord]],
) -> List[Dict[int, List[m.FaceRecord]]]:
    """[0] = finest, [-1] = root (single key). Each dict maps morton key -> merged face lists."""
    levels: List[Dict[int, List[m.FaceRecord]]] = [finest]
    cur = finest
    while len(cur) > 1:
        nxt: Dict[int, List[m.FaceRecord]] = defaultdict(list)
        for k, recs in cur.items():
            p = morton3d_parent_key(int(k))
            nxt[p].extend(recs)
        cur = dict(nxt)
        levels.append(cur)
    return levels


def _dfs_flat_nodes(
    level_maps: List[Dict[int, List[m.FaceRecord]]],
    *,
    root_level: int,
    root_key: int,
) -> List[_NodeBuild]:
    """Pre-order DFS; children are contiguous [child_begin, child_begin + child_count)."""

    buf: List[_NodeBuild] = []

    def visit(key: int, lod_level: int, parent_id: int) -> None:
        recs = level_maps[lod_level][key]
        my_id = len(buf)
        buf.append(
            _NodeBuild(
                parent=parent_id,
                level=lod_level,
                morton_key=int(key),
                records=recs,
                child_begin=-1,
                child_count=0,
            )
        )
        if lod_level == 0:
            return
        child_keys = [
            morton3d_child_key(int(key), o)
            for o in range(8)
            if morton3d_child_key(int(key), o) in level_maps[lod_level - 1]
        ]
        child_keys.sort()
        if not child_keys:
            return
        buf[my_id].child_begin = my_id + 1
        buf[my_id].child_count = len(child_keys)
        for ck in child_keys:
            visit(ck, lod_level - 1, my_id)

    visit(root_key, root_level, -1)
    return buf


def build_tree_and_export(
    mesh_path: Path,
    out_dir: Path,
    *,
    base_voxel_size: float = 0.01,
    use_sh: bool = False,
    bake_textures: bool = False,
    texture_samples: int = 256,
    ply_prefix: str = "lod",
) -> Path:
    if base_voxel_size <= 0:
        raise ValueError("base_voxel_size must be positive.")

    assert_material_mode_for_path(mesh_path, use_sh=use_sh)
    material_mode = cast(m.MaterialMode, "sh" if use_sh else "brdf")

    mesh = m.load_mesh(mesh_path)
    face_records = m.extract_face_records(
        mesh,
        material_mode=material_mode,
        texture_samples=texture_samples,
        bake_textures=bake_textures and use_sh,
    )
    if not face_records:
        raise ValueError("No valid face records extracted from mesh.")

    centers = np.stack([r.center for r in face_records], axis=0).astype(np.float64)
    ijk = m.voxel_index(centers, base_voxel_size)
    imin = ijk.min(axis=0).astype(np.int64)
    ijk_rel = ijk - imin
    max_extent = int(ijk_rel.max()) + 1
    bits_per_axis = max(1, int(np.ceil(np.log2(max(max_extent, 1)))))

    finest = _finest_voxel_morton(
        face_records, base_voxel_size, bits_per_axis=bits_per_axis, origin_ijk=imin
    )
    if not finest:
        raise ValueError("No voxels after Morton grouping (empty mesh?).")

    level_maps = _build_level_maps(finest)
    root_level = len(level_maps) - 1
    root_key = next(iter(level_maps[root_level]))

    flat = _dfs_flat_nodes(level_maps, root_level=root_level, root_key=root_key)
    n_nodes = len(flat)

    gaussians: List[m.GaussianPrimitive] = [
        m.fit_voxel_gaussian(node.records, material_mode=material_mode) for node in flat
    ]

    parent = np.array([n.parent for n in flat], dtype=np.int32)
    child_begin = np.array([n.child_begin for n in flat], dtype=np.int32)
    child_count = np.array([n.child_count for n in flat], dtype=np.int32)
    level = np.array([n.level for n in flat], dtype=np.uint16)
    morton_key = np.array([n.morton_key for n in flat], dtype=np.uint64)
    node_xyz = np.stack([g.xyz for g in gaussians], axis=0).astype(np.float32)

    log_scales = np.zeros((n_nodes, 3), dtype=np.float32)
    quat = np.zeros((n_nodes, 4), dtype=np.float32)
    for i, g in enumerate(gaussians):
        ls, q = m.covariance_to_log_scales_and_quat(g.covariance)
        log_scales[i] = ls.astype(np.float32)
        quat[i] = q.astype(np.float32)

    opacity = np.array([g.opacity for g in gaussians], dtype=np.float32)
    normals = np.stack([g.normal for g in gaussians], axis=0).astype(np.float32)

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_kw: Dict[str, np.ndarray] = {
        "parent": parent,
        "child_begin": child_begin,
        "child_count": child_count,
        "level": level,
        "morton_key": morton_key,
        "node_xyz": node_xyz,
        "pos": node_xyz.astype(np.float32),
        "opacity": opacity.reshape(-1, 1),
        "log_scales": log_scales,
        "rot": quat,
        "normals": normals,
        "base_voxel_size": np.array([base_voxel_size], dtype=np.float32),
        "root_level": np.array([root_level], dtype=np.int32),
        "bits_per_axis": np.array([bits_per_axis], dtype=np.int32),
        "voxel_origin_ijk": imin.astype(np.int32),
        "material_sh": np.array([1 if use_sh else 0], dtype=np.int8),
    }

    if use_sh:
        sh_dc = np.zeros((n_nodes, 3), dtype=np.float32)
        for i, g in enumerate(gaussians):
            if g.sh_color is None:
                raise ValueError("SH mode requires sh_color.")
            sh_dc[i] = ((g.sh_color - 0.5) / 0.28209).astype(np.float32)
        save_kw["sh_dc"] = sh_dc
    else:
        bc = np.zeros((n_nodes, 3), dtype=np.float32)
        mt = np.zeros(n_nodes, dtype=np.float32)
        rou = np.zeros(n_nodes, dtype=np.float32)
        for i, g in enumerate(gaussians):
            if g.basecolor_srgb is None or g.metallic is None or g.roughness is None:
                raise ValueError("BRDF mode requires basecolor, metallic, roughness.")
            bc[i] = g.basecolor_srgb.astype(np.float32)
            mt[i] = np.float32(g.metallic)
            rou[i] = np.float32(g.roughness)
        save_kw["basecolor"] = bc
        save_kw["metallic"] = mt
        save_kw["roughness"] = rou

    index_path = out_dir / "tree_index_flat.npz"
    np.savez_compressed(index_path, **save_kw)

    for L in range(root_level + 1):
        voxel_size_l = float(base_voxel_size * (2**L))
        mask = level == L
        idxs = np.flatnonzero(mask)
        sub = [gaussians[i] for i in idxs]
        out_ply = out_dir / f"{ply_prefix}_level_{L:02d}_vox{voxel_size_l:.6f}.ply"
        m.write_gaussian_ply(out_ply, sub, material_mode=material_mode)
        print(f"[level {L}] voxels={voxel_size_l:.6f}, nodes={len(sub)} -> {out_ply.name}")

    print(f"[tree] flat nodes={n_nodes}, root_level={root_level} -> {index_path}")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Morton octree + flat Gaussian table (child_begin/child_count); export per-level PLY."
    )
    parser.add_argument("--mesh", type=Path, required=True, help="Input mesh (OBJ/PLY/GLB/etc).")
    parser.add_argument(
        "--base-voxel-size",
        type=float,
        default=0.01,
        help="Finest voxel size (level 0).",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--sh",
        action="store_true",
        help="Spherical harmonics PLY; required for OBJ. Default is BRDF (GLB/GLTF).",
    )
    parser.add_argument(
        "--bake-textures",
        action="store_true",
        help="(SH only) Per-face texture average for color.",
    )
    parser.add_argument(
        "--texture-samples",
        type=int,
        default=256,
        help="Samples per face for BRDF or SH texture (default 256).",
    )
    parser.add_argument("--ply-prefix", type=str, default="lod", help="PLY filename prefix.")
    args = parser.parse_args()

    build_tree_and_export(
        mesh_path=args.mesh,
        out_dir=args.out_dir,
        base_voxel_size=args.base_voxel_size,
        use_sh=args.sh,
        bake_textures=args.bake_textures,
        texture_samples=args.texture_samples,
        ply_prefix=args.ply_prefix,
    )


if __name__ == "__main__":
    main()

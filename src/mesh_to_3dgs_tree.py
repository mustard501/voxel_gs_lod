from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, cast

import numpy as np

import mesh_to_3dgs_lod as m
from mesh_io import assert_material_mode_for_path
from morton import morton3d_child_key, morton3d_encode, morton3d_parent_key


EPS = 1e-8


@dataclass
class LODTree:
    base_voxel_size: float
    bbox_min: np.ndarray  # (3,)
    root_level: int

    # Per level, keys in the same order as written ply rows.
    level_keys: List[np.ndarray]  # level -> (Ni,)

    # Flattened global nodes: global_idx -> (level, key, xyz)
    node_level: np.ndarray  # (N,)
    node_key: np.ndarray  # (N,)
    node_xyz: np.ndarray  # (N,3)

    # Derived (built lazily)
    _level_key_to_global: Optional[List[Dict[int, int]]] = None

    def build_index(self) -> None:
        level_count = self.root_level + 1
        self._level_key_to_global = [dict() for _ in range(level_count)]
        for global_idx in range(self.node_level.shape[0]):
            l = int(self.node_level[global_idx])
            k = int(self.node_key[global_idx])
            self._level_key_to_global[l][k] = global_idx

    @staticmethod
    def load(tree_index_path: Path) -> "LODTree":
        d = np.load(tree_index_path, allow_pickle=False)

        base_voxel_size = float(d["base_voxel_size"][0])
        bbox_min = d["bbox_min"].astype(np.float32)
        root_level = int(d["root_level"][0])

        level_count = root_level + 1
        level_keys: List[np.ndarray] = []
        for l in range(level_count):
            level_keys.append(d[f"level_{l:02d}_keys"].astype(np.uint64))

        node_xyz = d["node_xyz"].astype(np.float32)
        node_key = d["node_key"].astype(np.uint64)
        node_level = d["node_level"].astype(np.uint16)

        return LODTree(
            base_voxel_size=base_voxel_size,
            bbox_min=bbox_min,
            root_level=root_level,
            level_keys=level_keys,
            node_level=node_level,
            node_key=node_key,
            node_xyz=node_xyz,
            _level_key_to_global=None,
        )

    def get_global_root_index(self) -> int:
        # Root key should be 0 (all coords shifted to 0 by construction).
        l = self.root_level
        k = 0
        if self._level_key_to_global is None:
            self.build_index()
        return self._level_key_to_global[l][k]

    def get_children_globals(self, parent_global_idx: int) -> List[int]:
        """
        Children of a node at level `l` are nodes at level `l-1` (one level finer).
        """

        if self._level_key_to_global is None:
            self.build_index()

        parent_level = int(self.node_level[parent_global_idx])
        if parent_level <= 0:
            return []

        child_level = parent_level - 1
        parent_key = int(self.node_key[parent_global_idx])
        out: List[int] = []
        for offset in range(8):
            child_key = morton3d_child_key(parent_key, offset)
            idx = self._level_key_to_global[child_level].get(child_key, None)
            if idx is not None:
                out.append(int(idx))
        return out

    def get_level_local_index(self, global_idx: int) -> tuple[int, int]:
        """
        Convert a global node index into:
          (level, local_row_index_in_that_level_ply)

        This assumes each level ply stores gaussians in ascending morton key order,
        which is how we wrote `level_keys` during build.
        """

        l = int(self.node_level[global_idx])
        k = int(self.node_key[global_idx])
        keys_l = self.level_keys[l]
        # keys_l is sorted => use binary search.
        local = int(np.searchsorted(keys_l, np.uint64(k), side="left"))
        if local >= keys_l.shape[0] or int(keys_l[local]) != k:
            raise KeyError(f"Global idx {global_idx} (level={l}, key={k}) not found in level_keys.")
        return l, local

    def cut_by_distance(
        self, camera_position: np.ndarray, *, tau: float = 0.05, max_nodes: Optional[int] = None
    ) -> List[int]:
        """
        Minimal real-time "cut": choose to refine children if the current voxel scale
        is still "large enough" for this node distance.

        Condition: (voxel_size_at_level / dist) > tau  => refine (down to finer nodes)
        """

        if self._level_key_to_global is None:
            self.build_index()

        cam = np.asarray(camera_position, dtype=np.float64).reshape(3)
        root_idx = self.get_global_root_index()

        render: List[int] = []
        stack: List[int] = [root_idx]

        while stack:
            node_idx = int(stack.pop())
            l = int(self.node_level[node_idx])
            d = float(np.linalg.norm(self.node_xyz[node_idx] - cam) + EPS)
            voxel_size = float(self.base_voxel_size * (2**l))

            # If we hit the finest level, we must render it.
            if l <= 0:
                render.append(node_idx)
                continue

            refine = (voxel_size / d) > tau
            if refine:
                children = self.get_children_globals(node_idx)
                if children:
                    stack.extend(children)
                    continue

            render.append(node_idx)
            if max_nodes is not None and len(render) >= max_nodes:
                break

        return render


def _fit_gaussian_from_faces(
    face_indices: np.ndarray,
    centers: np.ndarray,  # (F,3)
    areas: np.ndarray,  # (F,)
    normals_geom: np.ndarray,  # (F,3)
    vertices: np.ndarray,  # (F,3,3)
    *,
    material_mode: m.MaterialMode,
    colors: Optional[np.ndarray] = None,
    shading_normals: Optional[np.ndarray] = None,
    basecolors: Optional[np.ndarray] = None,
    metallic: Optional[np.ndarray] = None,
    roughness: Optional[np.ndarray] = None,
) -> m.GaussianPrimitive:
    face_indices = np.asarray(face_indices, dtype=np.int64)
    w = areas[face_indices].astype(np.float64)
    if w.size == 0:
        raise ValueError("Empty face set for gaussian fit.")

    xyz = m.weighted_average(centers[face_indices], w)
    covariance = m.pca_covariance_from_faces(vertices[face_indices], w)
    opacity = 1.0

    if material_mode == "sh":
        if colors is None:
            raise ValueError("SH fit requires colors array.")
        normal = m.weighted_average(normals_geom[face_indices], w)
        normal = normal / (np.linalg.norm(normal) + EPS)
        sh_color = m.weighted_average(colors[face_indices], w)
        sh_color = np.clip(sh_color, 0.0, 1.0)
        return m.GaussianPrimitive(
            xyz=xyz, normal=normal, covariance=covariance, opacity=opacity, sh_color=sh_color
        )

    if shading_normals is None or basecolors is None or metallic is None or roughness is None:
        raise ValueError("BRDF fit requires shading_normals, basecolors, metallic, roughness.")
    normal = m.weighted_average(shading_normals[face_indices], w)
    normal = normal / (np.linalg.norm(normal) + EPS)
    basecolor_srgb = m.weighted_average(basecolors[face_indices], w)
    basecolor_srgb = np.clip(basecolor_srgb, 0.0, 1.0)
    mt = metallic[face_indices]
    rf = roughness[face_indices]
    met = float(np.sum(mt * w) / (np.sum(w) + EPS))
    rou = float(np.sum(rf * w) / (np.sum(w) + EPS))
    met = float(np.clip(met, 0.0, 1.0))
    rou = float(np.clip(rou, 0.0, 1.0))
    return m.GaussianPrimitive(
        xyz=xyz,
        normal=normal,
        covariance=covariance,
        opacity=opacity,
        basecolor_srgb=basecolor_srgb,
        metallic=met,
        roughness=rou,
    )


def build_tree_and_export(
    mesh_path: Path,
    out_dir: Path,
    *,
    base_voxel_size: float = 0.01,
    use_sh: bool = False,
    bake_textures: bool = False,
    texture_samples: int = 256,
    ply_prefix: str = "lod",
) -> LODTree:
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
    areas = np.asarray([r.area for r in face_records], dtype=np.float64)
    normals_geom = np.stack([r.normal for r in face_records], axis=0).astype(np.float64)
    vertices = np.stack([r.vertices for r in face_records], axis=0).astype(np.float64)

    colors: Optional[np.ndarray] = None
    shading_normals: Optional[np.ndarray] = None
    basecolors: Optional[np.ndarray] = None
    metallic_arr: Optional[np.ndarray] = None
    roughness_arr: Optional[np.ndarray] = None
    if use_sh:
        colors = np.stack([cast(np.ndarray, r.color) for r in face_records], axis=0).astype(np.float64)
    else:
        shading_normals = np.stack([r.shading_normal for r in face_records], axis=0).astype(np.float64)
        basecolors = np.stack([r.basecolor_srgb for r in face_records], axis=0).astype(np.float64)
        metallic_arr = np.array([r.metallic for r in face_records], dtype=np.float64)
        roughness_arr = np.array([r.roughness for r in face_records], dtype=np.float64)

    bbox_min = centers.min(axis=0)
    rel = centers - bbox_min[None, :]

    i0 = np.floor(rel[:, 0] / base_voxel_size).astype(np.int64)
    j0 = np.floor(rel[:, 1] / base_voxel_size).astype(np.int64)
    k0 = np.floor(rel[:, 2] / base_voxel_size).astype(np.int64)

    max_i = int(i0.max()) if i0.size > 0 else 0
    max_j = int(j0.max()) if j0.size > 0 else 0
    max_k = int(k0.max()) if k0.size > 0 else 0
    root_level = max(max_i.bit_length(), max_j.bit_length(), max_k.bit_length())
    level_count = root_level + 1

    # 生成最细层级的morton code:key0
    bits_per_axis = max(max_i.bit_length(), max_j.bit_length(), max_k.bit_length())
    if bits_per_axis == 0:
        # Degenerate: all faces fall into one voxel cell.
        key0 = np.zeros(len(face_records), dtype=np.uint64)
    else:
        key0 = morton3d_encode(i0, j0, k0, bits_per_axis=bits_per_axis)

    out_dir.mkdir(parents=True, exist_ok=True)

    level_keys: List[np.ndarray] = []
    node_level_list: List[np.ndarray] = []
    node_key_list: List[np.ndarray] = []
    node_xyz_list: List[np.ndarray] = []

    global_offset = 0
    # 逐层向上构建
    for l in range(level_count):
        # 直接计算l层中祖先节点编码
        # 一定有很多重复元素，因为祖先节点在最细层会有多个子节点
        keys_l = key0 >> (3 * l)

        order = np.argsort(keys_l)
        keys_sorted = keys_l[order]

        # 计算出在排序后的keys_l中，
        # 每个编码对应的面片索引的范围
        if len(keys_sorted) == 0:
            unique_keys = np.array([], dtype=np.uint64)
            boundaries = np.array([], dtype=np.int64)
        else:
            change = np.ones(len(keys_sorted), dtype=bool)
            change[1:] = keys_sorted[1:] != keys_sorted[:-1]
            start_positions = np.nonzero(change)[0].astype(np.int64)
            unique_keys = keys_sorted[start_positions]

            end_positions = np.append(start_positions[1:], len(keys_sorted)).astype(np.int64)
            boundaries = np.stack([start_positions, end_positions], axis=1)

        gaussians: List[m.GaussianPrimitive] = []
        local_node_xyz: List[np.ndarray] = []
        local_node_keys: List[int] = []
        local_node_level: List[int] = []

        # 对key_l中每一种编码，即l层中每一个节点，依次用节点包含的面片拟合出一个高斯
        for node_i in range(unique_keys.shape[0]):
            s = int(boundaries[node_i, 0])
            e = int(boundaries[node_i, 1])
            face_idx = order[s:e]
            g = _fit_gaussian_from_faces(
                face_idx,
                centers=centers,
                areas=areas,
                normals_geom=normals_geom,
                vertices=vertices,
                material_mode=material_mode,
                colors=colors,
                shading_normals=shading_normals,
                basecolors=basecolors,
                metallic=metallic_arr,
                roughness=roughness_arr,
            )
            gaussians.append(g)
            # 计算第l层的局部信息
            local_node_xyz.append(g.xyz.astype(np.float32))
            local_node_keys.append(int(unique_keys[node_i]))
            local_node_level.append(l)

        # 将l层所有体素的morton key加入level_keys
        level_keys.append(unique_keys.astype(np.uint64))

        # 构建全局展平数据列表
        # 存入l层所有高斯的中心点坐标
        node_xyz_list.append(np.stack(local_node_xyz, axis=0) if local_node_xyz else np.zeros((0, 3), np.float32))
        # 存入这些高斯对应的morton key
        node_key_list.append(np.asarray(local_node_keys, dtype=np.uint64))
        # 存入 [l,l,...,l]
        node_level_list.append(np.asarray(local_node_level, dtype=np.uint16))

        # 导出第l层的高斯模型
        voxel_size_l = base_voxel_size * (2**l)
        ply_path = out_dir / f"{ply_prefix}_level_{l:02d}_vox{voxel_size_l:.6f}.ply"
        m.write_gaussian_ply(ply_path, gaussians, material_mode=material_mode)
        print(f"[Tree] level={l}/{root_level}, nodes={len(gaussians)} -> {ply_path}")
        # 当前生成高斯数
        global_offset += len(gaussians)

    node_xyz = np.concatenate(node_xyz_list, axis=0) if node_xyz_list else np.zeros((0, 3), np.float32)
    node_key = np.concatenate(node_key_list, axis=0) if node_key_list else np.zeros((0,), np.uint64)
    node_level = np.concatenate(node_level_list, axis=0) if node_level_list else np.zeros((0,), np.uint16)

    # 实例化LODTree
    tree = LODTree(
        base_voxel_size=float(base_voxel_size),
        bbox_min=bbox_min.astype(np.float32),
        root_level=int(root_level),
        level_keys=level_keys,
        node_level=node_level,
        node_key=node_key,
        node_xyz=node_xyz,
    )

    # 以字典形式存储
    tree_index_path = out_dir / "tree_index.npz"
    save_dict = {
        "base_voxel_size": np.array([tree.base_voxel_size], dtype=np.float32),
        "bbox_min": tree.bbox_min,
        "root_level": np.array([tree.root_level], dtype=np.int32),
        "node_xyz": tree.node_xyz.astype(np.float32),
        "node_key": tree.node_key.astype(np.uint64),
        "node_level": tree.node_level.astype(np.uint16),
    }
    # 将每一层独立的有序 Key 数组以 level_00_keys, level_01_keys 等名称存入字典
    for l, keys in enumerate(tree.level_keys):
        save_dict[f"level_{l:02d}_keys"] = keys.astype(np.uint64)
    np.savez_compressed(tree_index_path, **save_dict)

    print(f"[Tree] saved index -> {tree_index_path}")
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Build voxel-based 3DGS LOD tree (Morton indexed).")
    parser.add_argument("--mesh", type=Path, required=True, help="Input mesh (OBJ/PLY/GLB/etc).")
    parser.add_argument(
        "--base-voxel-size",
        type=float,
        default=0.01,
        help="Finest voxel size for LOD level 0 (e.g., 0.01).",
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
        help="Samples per face: BRDF basecolor/normal integration or SH texture (default 256).",
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


from __future__ import annotations

from pathlib import Path

import trimesh

"""
加载时封装，统一加载为trimesh.Trimesh对象
"""
def is_obj_path(path: Path) -> bool:
    return path.suffix.lower() == ".obj"


def is_gltf_path(path: Path) -> bool:
    return path.suffix.lower() in (".glb", ".gltf")


def load_mesh_uniform(mesh_path: Path) -> trimesh.Trimesh:
    """Load geometry as a single Trimesh (OBJ / GLB / GLTF / …)."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Input is not a triangle mesh: {mesh_path}")
    if mesh.faces.shape[0] == 0:
        raise ValueError("Mesh has no faces.")
    if mesh.vertices.shape[0] == 0:
        raise ValueError("Mesh has no vertices.")
    return mesh


def assert_material_mode_for_path(mesh_path: Path, *, use_sh: bool) -> None:
    """
    BRDF/PBR 默认仅支持 GLB/GLTF；OBJ 仅允许球谐管线（--sh）。
    """
    if use_sh:
        return
    if is_obj_path(mesh_path):
        raise ValueError(
            "Default BRDF export requires GLB or GLTF. For OBJ use --sh (spherical harmonics), "
            "or convert the asset to GLB."
        )

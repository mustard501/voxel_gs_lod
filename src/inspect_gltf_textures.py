#!/usr/bin/env python3
"""
检查 GLB / GLTF 中各几何体与材质的纹理与因子（PBR / 简单材质）。

用法:
  python src/inspect_gltf_textures.py path/to/model.glb
  python src/inspect_gltf_textures.py path/to/model.gltf --mesh
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial, SimpleMaterial


def _img_desc(img: Any) -> str:
    if img is None:
        return "(无)"
    try:
        w, h = img.size
        mode = getattr(img, "mode", "?")
        return f"{w}x{h} {mode}"
    except Exception as e:
        return f"(无法读取: {e})"


def _fmt_array(x: Any) -> str:
    if x is None:
        return "(无)"
    a = np.asarray(x, dtype=np.float64).ravel()
    return np.array2string(a, precision=4, separator=", ", max_line_width=120)


def _report_pbr(name: str, mat: PBRMaterial) -> None:
    print(f"  [PBR] name={mat.name!r}")
    print(f"        baseColorTexture:     {_img_desc(mat.baseColorTexture)}")
    print(f"        baseColorFactor:      {_fmt_array(mat.baseColorFactor)}")
    print(f"        metallicRoughnessTex: {_img_desc(mat.metallicRoughnessTexture)}")
    print(f"        metallicFactor:       {_fmt_array(mat.metallicFactor)}")
    print(f"        roughnessFactor:      {_fmt_array(mat.roughnessFactor)}")
    print(f"        normalTexture:        {_img_desc(mat.normalTexture)}")
    print(f"        occlusionTexture:     {_img_desc(mat.occlusionTexture)}")
    print(f"        emissiveTexture:      {_img_desc(mat.emissiveTexture)}")
    print(f"        emissiveFactor:       {_fmt_array(mat.emissiveFactor)}")
    print(f"        alphaMode:            {mat.alphaMode!r}")
    print(f"        alphaCutoff:          {mat.alphaCutoff!r}")
    print(f"        doubleSided:          {mat.doubleSided!r}")


def _report_simple(name: str, mat: SimpleMaterial) -> None:
    print(f"  [SimpleMaterial] name={mat.name!r}")
    print(f"        image (diffuse):      {_img_desc(mat.image)}")
    print(f"        diffuse:              {_fmt_array(mat.diffuse)}")


def _mesh_visual_summary(mesh: trimesh.Trimesh, label: str) -> None:
    vis = mesh.visual
    print(f"\n--- {label} ---")
    print(f"  vertices={len(mesh.vertices):,}  faces={len(mesh.faces):,}")
    if vis is None:
        print("  visual: (无)")
        return
    kind = getattr(vis, "kind", None)
    print(f"  visual.kind: {kind!r}")
    uv = getattr(vis, "uv", None)
    if uv is not None:
        print(f"  uv: shape={uv.shape} dtype={uv.dtype}")
    else:
        print("  uv: (无)")
    mat = getattr(vis, "material", None)
    if mat is None:
        print("  material: (无)")
        return
    if isinstance(mat, PBRMaterial):
        _report_pbr(label, mat)
    elif isinstance(mat, SimpleMaterial):
        _report_simple(label, mat)
    else:
        print(f"  material: {type(mat).__name__} (未逐项展开)")


def inspect_scene(path: Path, *, also_dump_mesh: bool) -> None:
    scene = trimesh.load(path, force="scene")
    if not isinstance(scene, trimesh.Scene):
        raise SystemExit(f"预期为 Scene，得到: {type(scene).__name__}")

    print(f"文件: {path.resolve()}")
    print(f"Scene 几何体数量: {len(scene.geometry)}")

    for geom_name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            _mesh_visual_summary(geom, f"geometry[{geom_name!r}]")
        else:
            print(f"\n--- geometry[{geom_name!r}] ---")
            print(f"  类型: {type(geom).__name__}（非 Trimesh，跳过纹理详情）")

    if also_dump_mesh:
        print("\n========== force='mesh' 合并后（与部分导出管线一致）==========")
        merged = trimesh.load(path, force="mesh")
        if isinstance(merged, trimesh.Trimesh):
            _mesh_visual_summary(merged, "merged Trimesh")
        else:
            print(f"  合并结果类型: {type(merged).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 GLB/GLTF 纹理与 PBR 属性。")
    parser.add_argument("path", type=Path, help="输入 .glb 或 .gltf")
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="额外打印 trimesh.load(force='mesh') 合并后的材质/纹理（便于对照 LOD 脚本）",
    )
    args = parser.parse_args()

    p = args.path
    if not p.is_file():
        raise SystemExit(f"文件不存在: {p}")
    suf = p.suffix.lower()
    if suf not in (".glb", ".gltf"):
        raise SystemExit("请使用 .glb 或 .gltf 文件")

    inspect_scene(p, also_dump_mesh=args.mesh)


if __name__ == "__main__":
    main()

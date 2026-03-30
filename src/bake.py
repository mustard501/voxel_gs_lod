from __future__ import annotations

import trimesh
import numpy as np
from tqdm import tqdm
from trimesh.visual import color as visual_color

"""
如果 mesh 只有贴图
方法一:优先使用 `texture_face_mean_colors`：在 UV 三角上均匀采样并双线性取色，
得到每个面片在贴图上的平均颜色（对纹理测度的积分近似）。
"""

def _material_main_image(mesh: trimesh.Trimesh):
    vis = mesh.visual
    if vis is None or getattr(vis, "kind", None) != "texture":
        return None
    mat = getattr(vis, "material", None)
    if mat is None:
        return None
    img = getattr(mat, "image", None)
    if img is None:
        img = getattr(mat, "baseColorTexture", None)
    return img


def texture_face_mean_colors(
    mesh: trimesh.Trimesh,
    faces: np.ndarray,
    n_samples: int = 256,
    *,
    seed: int = 0,
    max_uv_per_batch: int = 262_144,
    show_progress: bool = True,
) -> np.ndarray | None:
    """
    对每个三角面片，在 UV 平面上均匀随机取点，双线性采样纹理，取平均作为该面片颜色。
    等价于在 3D 三角上以面积均匀分布采样（仿射 UV 下与 UV 三角上均匀分布一一对应）。

    大模型面片很多时按批调用 trimesh 采样，避免一次性分配 F×S 条 UV 导致内存爆炸。
    """
    if n_samples <= 0:
        return None
    image = _material_main_image(mesh)
    uv_all = getattr(mesh.visual, "uv", None)
    if image is None or uv_all is None or len(uv_all) != len(mesh.vertices):
        return None

    faces = np.asarray(faces, dtype=np.int64)
    uv_tri = uv_all[faces].astype(np.float64, copy=False)  # (F, 3, 2)
    f_count = uv_tri.shape[0]
    if f_count == 0:
        return None

    rng = np.random.default_rng(seed)
    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)
    sr = np.sqrt(r1)
    l1 = 1.0 - sr
    l2 = sr * (1.0 - r2)
    l3 = sr * r2

    face_batch = max(1, max_uv_per_batch // n_samples)
    mean_rgb = np.zeros((f_count, 3), dtype=np.float64)

    batch_starts = range(0, f_count, face_batch)
    for start in tqdm(
        batch_starts,
        desc="纹理面片采样",
        unit="批",
        disable=not show_progress,
    ):
        end = min(start + face_batch, f_count)
        uv_b = uv_tri[start:end]
        b = end - start
        uv_samples = (
            l1[np.newaxis, :, np.newaxis] * uv_b[:, 0:1, :]
            + l2[np.newaxis, :, np.newaxis] * uv_b[:, 1:2, :]
            + l3[np.newaxis, :, np.newaxis] * uv_b[:, 2:3, :]
        )
        flat = uv_samples.reshape(-1, 2)
        rgba = visual_color.uv_to_interpolated_color(flat, image)
        if rgba is None:
            return None
        rgb = np.asarray(rgba[:, :3], dtype=np.float64) / 255.0
        mean_rgb[start:end] = rgb.reshape(b, n_samples, 3).mean(axis=1)

    return mean_rgb

"""
方法二:将纹理bake到面片顶点上
"""
def bake_texture_to_vertices(mesh_path):
    # 1. 加载模型 
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # 2. 检查是否有材质和 UV 坐标
    if not hasattr(mesh.visual, 'uv'):
        print("模型没有 UV 坐标，无法烘焙贴图。")
        return mesh

    # 3. 使用 trimesh 内置的 color_mappers 将纹理采样为顶点颜色
    v_colors = mesh.visual.to_color().vertex_colors
    
    # 4. 创建一个新的 mesh 或更新现有 mesh 的视觉属性
    new_mesh = mesh.copy()
    new_mesh.visual = trimesh.visual.ColorVisuals(
        mesh=new_mesh, 
        vertex_colors=v_colors
    )
    print(v_colors.max(), v_colors.min())
    return new_mesh

def bake_to_ply():
    baked_mesh = bake_texture_to_vertices("assets/inputs/tree/tree.obj")
    baked_mesh.export("baked_model.ply")
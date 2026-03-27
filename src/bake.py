import trimesh
import numpy as np

"""
如果mesh只有贴图，先将贴图bake进mesh的顶点颜色里
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

# if __name__ == "__main__":
def bake_to_ply():
    baked_mesh = bake_texture_to_vertices("assets/inputs/tree/tree.obj")
    baked_mesh.export("baked_model.ply")
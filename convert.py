import numpy as np
from plyfile import PlyData, PlyElement
import os

def convert_custom_ply_to_visuals(input_path):
    print(f"正在读取: {input_path}")
    plydata = PlyData.read(input_path)
    v = plydata['vertex']
    num_verts = len(v)

    # 1. 定义转换逻辑
    # 显式处理单列和多列的情况
    tasks = [
        ("albedo", ['base_r', 'base_g', 'base_b'], lambda x: x),
        ("normal", ['nx', 'ny', 'nz'], lambda x: 0.5 * x + 0.5),
        ("roughness", ['roughness'], lambda x: np.column_stack([x, x, x])),
        ("metallic", ['metallic'], lambda x: np.column_stack([x, x, x]))
    ]

    base_name = os.path.splitext(input_path)[0]

    for suffix, src_cols, transform in tasks:
        print(f"正在生成 {suffix} 版本...")
        
        # 提取数据：如果是多列，stack 成 (N, 3)；如果是单列，保持 (N,)
        if len(src_cols) > 1:
            src_data = np.stack([v[col] for col in src_cols], axis=1)
        else:
            src_data = v[src_cols[0]]
        
        src_data = (src_data-0.5)/0.28209

        # 执行转换：确保结果一定是 (N, 3)
        fake_dc = transform(src_data).astype(np.float32)

        # 构造属性列表
        vertex_data = []
        # 标准坐标
        for p in ['x', 'y', 'z']:
            vertex_data.append((v[p], p))
        
        # 写入伪装的 f_dc (关键修复点)
        vertex_data.append((fake_dc[:, 0], 'f_dc_0'))
        vertex_data.append((fake_dc[:, 1], 'f_dc_1'))
        vertex_data.append((fake_dc[:, 2], 'f_dc_2'))

        # 其他 3DGS 必需属性
        other_props = ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        for p in other_props:
            if p in v.data.dtype.names:
                vertex_data.append((v[p], p))

        # 构造 Numpy 结构化数组
        names = [d[1] for d in vertex_data]
        formats = ['f4'] * len(names)
        composite_array = np.empty(num_verts, dtype={'names': names, 'formats': formats})
        
        for data, name in vertex_data:
            composite_array[name] = data

        new_el = PlyElement.describe(composite_array, 'vertex')
        output_path = f"{base_name}_{suffix}.ply"
        PlyData([new_el], text=False).write(output_path)
        print(f"成功保存: {output_path}")

if __name__ == "__main__":
    # 替换为你的文件路径
    input_file = "assets/outputs/2/lod_level_00_vox0.010000.ply" 
    if os.path.exists(input_file):
        convert_custom_ply_to_visuals(input_file)
    else:
        print("未找到输入文件，请检查路径。")
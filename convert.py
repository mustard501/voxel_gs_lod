import numpy as np
from plyfile import PlyData, PlyElement
import os
from pathlib import Path

def process_directory(input_dir):
    # 1. 准备基础路径
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"错误: {input_dir} 不是一个有效的目录")
        return

    # 2. 定义任务和对应的子文件夹
    tasks = [
        ("albedo", ['base_r', 'base_g', 'base_b'], lambda x: x),
        ("normal", ['nx', 'ny', 'nz'], lambda x: 0.5 * x + 0.5),
        ("roughness", ['roughness'], lambda x: np.column_stack([x, x, x])),
        ("metallic", ['metallic'], lambda x: np.column_stack([x, x, x]))
    ]

    # 3. 创建输出文件夹
    output_dirs = {}
    for task_name, _, _ in tasks:
        folder = input_path / task_name
        folder.mkdir(parents=True, exist_ok=True)
        output_dirs[task_name] = folder

    # 4. 遍历目录下所有的 ply 文件
    ply_files = list(input_path.glob("*.ply"))
    if not ply_files:
        print("未在该目录下找到 .ply 文件。")
        return

    print(f"找到 {len(ply_files)} 个文件，开始处理...")

    for ply_file in ply_files:
        print(f"\n--- 正在处理: {ply_file.name} ---")
        
        try:
            plydata = PlyData.read(str(ply_file))
            v = plydata['vertex']
            num_verts = len(v)

            for suffix, src_cols, transform in tasks:
                # 检查输入列是否存在
                if not all(col in v.data.dtype.names for col in src_cols):
                    print(f"跳过 {suffix}: 缺少属性 {src_cols}")
                    continue

                # 提取数据
                if len(src_cols) > 1:
                    src_data = np.stack([v[col] for col in src_cols], axis=1)
                else:
                    src_data = v[src_cols[0]]
                
                # 3DGS 特有的 DC 系数转换逻辑
                src_data = (src_data - 0.5) / 0.28209

                fake_dc = transform(src_data).astype(np.float32)

                vertex_data = []
                for p in ['x', 'y', 'z']:
                    vertex_data.append((v[p], p))
                
                # 写入伪装的 f_dc_0, 1, 2
                vertex_data.append((fake_dc[:, 0], 'f_dc_0'))
                vertex_data.append((fake_dc[:, 1], 'f_dc_1'))
                vertex_data.append((fake_dc[:, 2], 'f_dc_2'))

                # 复制其他 3DGS 必需属性
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

                # 生成新文件
                new_el = PlyElement.describe(composite_array, 'vertex')
                # 保持原文件名，存入对应的子文件夹
                output_path = output_dirs[suffix] / ply_file.name
                
                PlyData([new_el], text=False).write(str(output_path))
                print(f"已生成: {suffix} -> {output_path.name}")

        except Exception as e:
            print(f"处理文件 {ply_file.name} 时出错: {e}")

if __name__ == "__main__":
    # 在这里输入你的目标文件夹路径
    target_directory = "assets/outputs/3" 
    
    if os.path.exists(target_directory):
        process_directory(target_directory)
    else:
        print("目录不存在，请检查路径。")
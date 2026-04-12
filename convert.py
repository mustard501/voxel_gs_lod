import os
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

# 3DGS 球谐 DC 与 RGB 的常用换算系数（与 Inria 3DGS 一致）
_SH_DC_C0 = 0.28209479177387814


def srgb_to_linear_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    sRGB 编码的 RGB（每通道通常 0–1）转为线性光 RGB。
    IEC 61966-2-1 近似分段。
    """
    c = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    low = c <= 0.04045
    return np.where(low, c / 12.92, np.power((c + 0.055) / 1.055, 2.4)).astype(np.float64)


def linear_to_srgb_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    线性光 RGB 转为 sRGB 编码（0–1）。
    """
    c = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    low = c <= 0.0031308
    out = np.where(low, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return np.clip(out, 0.0, 1.0).astype(np.float64)


def _rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """线性或「按通道直接当 SH 输入」的 RGB -> f_dc = (rgb - 0.5) / C0。"""
    return (np.asarray(rgb, dtype=np.float64) - 0.5) / _SH_DC_C0


def process_directory(input_dir):
    # 1. 准备基础路径
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"错误: {input_dir} 不是一个有效的目录")
        return

    # 2. 非 albedo 任务：与原先一致
    other_tasks = [
        ("normal", ["nx", "ny", "nz"], lambda x: 0.5 * x + 0.5),
        ("roughness", ["roughness"], lambda x: np.column_stack([x, x, x])),
        ("metallic", ["metallic"], lambda x: np.column_stack([x, x, x])),
    ]

    albedo_base_cols = ["base_r", "base_g", "base_b"]
    # albedo_srgb：PLY 内 basecolor 原样参与球谐换算（兼容旧「albedo」行为）
    # albedo_rgb：先 sRGB -> 线性 RGB，再按线性值做球谐换算（推荐与物理渲染一致）
    albedo_modes = [
        ("albedo_srgb", lambda base: base),
        ("albedo_rgb", srgb_to_linear_rgb),
    ]

    # 3. 创建输出子文件夹
    output_dir_names = [name for name, _, _ in other_tasks] + [name for name, _ in albedo_modes]
    output_dirs = {}
    for name in output_dir_names:
        folder = input_path / name
        folder.mkdir(parents=True, exist_ok=True)
        output_dirs[name] = folder

    # 4. 遍历目录下所有的 ply 文件（与旧逻辑一致：ply 在 input_dir 根下）
    ply_files = list(input_path.glob("*.ply"))
    if not ply_files:
        print("未在该目录下找到 .ply 文件。")
        return

    print(f"找到 {len(ply_files)} 个文件，开始处理...")

    for ply_file in ply_files:
        print(f"\n--- 正在处理: {ply_file.name} ---")

        try:
            plydata = PlyData.read(str(ply_file))
            v = plydata["vertex"]
            num_verts = len(v)

            # ----- albedo_srgb / albedo_rgb -----
            if all(col in v.data.dtype.names for col in albedo_base_cols):
                base = np.stack([v[col] for col in albedo_base_cols], axis=1).astype(
                    np.float64, copy=False
                )
                for folder_name, to_sh_rgb in albedo_modes:
                    rgb_for_sh = to_sh_rgb(base)
                    src_data = _rgb_to_sh_dc(rgb_for_sh)
                    fake_dc = src_data.astype(np.float32)

                    vertex_data = []
                    for p in ["x", "y", "z"]:
                        vertex_data.append((v[p], p))
                    vertex_data.append((fake_dc[:, 0], "f_dc_0"))
                    vertex_data.append((fake_dc[:, 1], "f_dc_1"))
                    vertex_data.append((fake_dc[:, 2], "f_dc_2"))

                    other_props = [
                        "opacity",
                        "scale_0",
                        "scale_1",
                        "scale_2",
                        "rot_0",
                        "rot_1",
                        "rot_2",
                        "rot_3",
                    ]
                    for p in other_props:
                        if p in v.data.dtype.names:
                            vertex_data.append((v[p], p))

                    names = [d[1] for d in vertex_data]
                    composite_array = np.empty(num_verts, dtype={"names": names, "formats": ["f4"] * len(names)})
                    for data, name in vertex_data:
                        composite_array[name] = data

                    new_el = PlyElement.describe(composite_array, "vertex")
                    output_path = output_dirs[folder_name] / ply_file.name
                    PlyData([new_el], text=False).write(str(output_path))
                    print(f"已生成: {folder_name} -> {output_path.name}")
            else:
                print(f"跳过 albedo_*: 缺少属性 {albedo_base_cols}")

            # ----- normal / roughness / metallic -----
            for suffix, src_cols, transform in other_tasks:
                if not all(col in v.data.dtype.names for col in src_cols):
                    print(f"跳过 {suffix}: 缺少属性 {src_cols}")
                    continue

                if len(src_cols) > 1:
                    src_data = np.stack([v[col] for col in src_cols], axis=1)
                else:
                    src_data = v[src_cols[0]]

                src_data = (src_data - 0.5) / _SH_DC_C0
                fake_dc = transform(src_data).astype(np.float32)

                vertex_data = []
                for p in ["x", "y", "z"]:
                    vertex_data.append((v[p], p))
                vertex_data.append((fake_dc[:, 0], "f_dc_0"))
                vertex_data.append((fake_dc[:, 1], "f_dc_1"))
                vertex_data.append((fake_dc[:, 2], "f_dc_2"))

                other_props = [
                    "opacity",
                    "scale_0",
                    "scale_1",
                    "scale_2",
                    "rot_0",
                    "rot_1",
                    "rot_2",
                    "rot_3",
                ]
                for p in other_props:
                    if p in v.data.dtype.names:
                        vertex_data.append((v[p], p))

                names = [d[1] for d in vertex_data]
                composite_array = np.empty(num_verts, dtype={"names": names, "formats": ["f4"] * len(names)})
                for data, name in vertex_data:
                    composite_array[name] = data

                new_el = PlyElement.describe(composite_array, "vertex")
                output_path = output_dirs[suffix] / ply_file.name
                PlyData([new_el], text=False).write(str(output_path))
                print(f"已生成: {suffix} -> {output_path.name}")

        except Exception as e:
            print(f"处理文件 {ply_file.name} 时出错: {e}")


if __name__ == "__main__":
    target_directory = "assets/outputs/bunny"

    if os.path.exists(target_directory):
        process_directory(target_directory)
    else:
        print("目录不存在，请检查路径。")

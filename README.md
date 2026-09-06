# Voxel_GS_LOD

将三角网格模型按多尺度体素进行采样，并导出多层高斯点云（`.ply`）。

## 导出模式（默认 BRDF，球谐用 `--sh`）

| 模式 | 输入 | PLY 属性 |
|------|------|----------|
| **BRDF（默认）** | **GLB / GLTF**（glTF PBR 材质） | `base_r/g/b`（sRGB 0–1）、`nx/y/z`（着色法线）、`metallic`、`roughness`，以及尺度/旋转/不透明 |
| **球谐 `--sh`** | **OBJ** 等任意三角网格 | 经典 3DGS：`f_dc_0/1/2`（SH DC）、几何法线等 |

- **OBJ 不能使用默认 BRDF**；请对 OBJ 加 `--sh`，或先转成 GLB。
- BRDF：`basecolor` 与 **normal map → 世界空间着色法线** 均在 UV 三角上均匀采样做积分近似；`metallic` / `roughness` 在面片 **UV 重心** 双线性取样（无贴图时用材质因子）。

## 方法概述

- 对 mesh 面片计算中心、法线、面积；BRDF 另算 basecolor、着色法线、金属度、粗糙度。
- 按体素尺寸将面片中心分桶，每个体素拟合一个高斯（位置、法线、颜色或 PBR 量均为**面积加权**；协方差为加权 PCA）。

### SH 路径与贴图（`--sh`）

- `--bake-textures`：在 UV 上均匀采样做面片平均色（积分近似）；`--texture-samples` 控制采样数（默认 256）。
- 未开 `--bake-textures` 时用顶点色 / 面片色 / 灰。

### 顶点 bake（独立工具）

`src/bake.py` 中 `bake_texture_to_vertices` 可将贴图 bake 到顶点并导出 PLY，与主 LOD 管线独立。

## 安装

```bash
pip install -r requirements.txt
```

## 使用（多尺度 LOD）

**默认：GLB + BRDF**

```bash
python src/mesh_to_3dgs_lod.py --mesh path/to/model.glb --voxel-sizes 0.01,0.02,0.04 --out-dir path/to/output_dir
```

**OBJ + 球谐**

```bash
python src/mesh_to_3dgs_lod.py --mesh path/to/model.obj --voxel-sizes 0.01,0.02,0.04 --out-dir path/to/output_dir --sh
```

带贴图平均色（仅 SH）：

```bash
python src/mesh_to_3dgs_lod.py ... --sh --bake-textures
```

可选：`--texture-samples 512`（BRDF 下用于 basecolor/normal 积分；SH 下用于 `--bake-textures`）。

---

## LOD 层次树（Morton）

**默认 BRDF（GLB）**

```bash
python src/mesh_to_3dgs_tree.py --mesh path/to/model.glb --base-voxel-size 0.01 --out-dir path/to/output_dir
```

**球谐（OBJ）**

```bash
python src/mesh_to_3dgs_tree.py --mesh path/to/model.obj --base-voxel-size 0.01 --out-dir path/to/output_dir --sh --bake-textures
```

输出：每层 `lod_level_XX_vox*.ply`，以及 `tree_index.npz`。

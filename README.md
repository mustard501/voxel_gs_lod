# Voxel_GS_LOD

将三角网格按体素采样，拟合多层 3D Gaussian 点云（`.ply`）。支持手动多尺度 LOD 与 Morton 层级树两条管线。

## 导出模式

| 模式 | 输入 | PLY 属性 |
|------|------|----------|
| **BRDF（默认）** | GLB / GLTF（PBR） | `base_r/g/b`、`nx/y/z`（着色法线）、`metallic`、`roughness`、尺度/旋转/不透明 |
| **球谐 `--sh`** | OBJ 等任意网格 | `f_dc_0/1/2`、几何法线 `nx/y/z`、尺度/旋转/不透明 |

- OBJ 须加 `--sh`，或先转 GLB。
- BRDF：basecolor 与 normal map 在 UV 三角上积分；metallic/roughness 在 UV 重心采样（无贴图用材质因子）。
- SH：`--bake-textures` 做贴图面片平均色；否则用顶点色/面片色/灰。`--texture-samples` 默认 256。

BRDF PLY 字段详见 [docs/mesh_to_3dgs_tree_ply_params.md](docs/mesh_to_3dgs_tree_ply_params.md)。

## 安装

```bash
pip install -r requirements.txt
```

查看器（可选，需 CUDA）：

```bash
pip install torch
pip install -e submodules
```

---

## 工作流

### 1. 手动多尺度 LOD

**入口** `src/mesh_to_3dgs_lod.py` — 指定若干体素尺寸，各导出一份 PLY。

| | BRDF | 球谐 |
|---|------|------|
| 命令 | `python src/mesh_to_3dgs_lod.py --mesh model.glb --voxel-sizes 0.01,0.02,0.04 --out-dir out/` | 同上 + `--sh`；贴图加 `--bake-textures` |
| 输出 | `out/lod_00_vox0.0100.ply`, `lod_01_vox0.0200.ply`, … | 同目录，属性为 SH |

### 2. Morton 层级树（推荐）

**入口** `src/tree_generator.py` — 从最细体素自动向上合并，输出层级 PLY 与 flat 树索引。

| | BRDF | 球谐 |
|---|------|------|
| 命令 | `python src/tree_generator.py --mesh model.glb --base-voxel-size 0.01 --out-dir out/` | 同上 + `--sh --bake-textures` |
| 输出 | `out/lod_level_00_vox0.010000.ply`, … | 同目录 |
| 索引 | `out/tree_index_flat.npz`（`child_begin` / `child_count` 层级遍历） | 同 |

可选 `--ply-prefix` 改 PLY 文件名前缀（默认 `lod`）。

### 3. BRDF 属性转 SH-DC 视图（可选）

**入口** `convert.py` — 将目录内 BRDF `.ply` 转为可光栅化的多通道 SH-DC PLY。

- **输入**：含 `base_r/g/b` 等 BRDF 属性的 `.ply`
- **输出**（输入目录下的子文件夹）：`albedo_srgb/`、`albedo_rgb/`、`normal/`、`metallic/`、`roughness/`

需在脚本内设置 `target_directory`（默认 `assets/outputs/city`）。

### 4. LOD 渲染查看（可选，仅 BRDF 树）

**入口** `scripts/lod_flat_dgr_viewer.py` — 基于相机距离的 LOD 切换，切换 basecolor / normal / metallic / roughness 视图。

```bash
python scripts/lod_flat_dgr_viewer.py --lod-dir out/
```

- **输入**：`tree_index_flat.npz` + `lod_level_*_vox*.ply`（由步骤 2 生成）
- **依赖**：CUDA、`submodules` 已编译安装

---

## 辅助工具

- `src/inspect_gltf_textures.py` — 检查 GLB/GLTF 纹理与 PBR 因子
- `plyheader.py` — 查看 PLY 头与属性列表

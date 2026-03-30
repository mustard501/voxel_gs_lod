# Mesh to 3DGS LOD

将三角网格模型按多尺度体素进行采样，并导出多层 3D Gaussian Splatting (`.ply`)。

## 方法概述

- 对 mesh 面片计算中心、法线、面积、颜色。
- 按给定体素尺寸将面片中心分桶。
- 每个体素拟合一个高斯：
  - 位置：面片中心按面积加权平均。
  - 法线：面片法线按面积加权平均后归一化。
  - 颜色（SH 的 DC 项）：体素内各面片颜色按**面片面积**加权平均。
  - 协方差：体素内面片顶点做加权 PCA 得到。
- 不同体素尺寸对应不同 LOD。

### 带贴图（UV）的模型

- 使用 `--bake-textures` 时，**不会**再把贴图 bake 到顶点色；而是在每个三角面的 UV 三角上**均匀采样**，对纹理做**双线性插值**取色，再对样本取平均，作为该面在贴图上的平均颜色（对表面颜色的面积积分近似）。
- 可用 `--texture-samples` 指定每个面的采样数（默认 `256`），在精度与耗时之间权衡。
- 若未开 `--bake-textures`，则优先使用 mesh 自带的顶点色 / 面片色；都没有时使用中性灰。

### 单独导出「顶点已 bake 贴图」的 mesh

脚本 `src/bake.py` 中的 `bake_texture_to_vertices` 仍可将贴图转为顶点色并导出 PLY，供其它工具使用；与上述 3DGS 导出管线独立。

## 安装

```bash
pip install -r requirements.txt
```

## 使用（多尺度 LOD）

```bash
python src/mesh_to_3dgs_lod.py --mesh path/to/model.obj --voxel-sizes 0.01,0.02,0.04 --out-dir path/to/output_dir
```

带 UV 贴图时启用按面片纹理平均色：

```bash
python src/mesh_to_3dgs_lod.py --mesh path/to/model.obj --voxel-sizes 0.01,0.02,0.04 --out-dir path/to/output_dir --bake-textures
```

可选：提高每面采样数（例如 512）：

```bash
--texture-samples 512
```

输出示例：

- `output_lods/lod_00_vox0.0100.ply`
- `output_lods/lod_01_vox0.0200.ply`
- `output_lods/lod_02_vox0.0400.ply`

---

## 生成 LOD 层次树（每层重新拟合 + Morton 索引）

```bash
python src/mesh_to_3dgs_tree.py --mesh path/to/model.obj --base-voxel-size 0.01 --out-dir path/to/output_dir --bake-textures
```

同样支持 `--texture-samples`。树构建中每个节点的高斯颜色也是其包含面片颜色的**面积加权**平均。

输出：

- 每层一个 ply：`output_tree/lod_level_XX_vox*.ply`
- 树索引文件：`output_tree/tree_index.npz`

# Mesh to 3DGS LOD

将三角网格模型按多尺度体素进行采样，并导出多层 3D Gaussian Splatting (`.ply`)。

## 方法概述

- 对 mesh 面片计算中心、法线、面积、颜色。
- 按给定体素尺寸将面片中心分桶。
- 每个体素拟合一个高斯：
  - 位置：面片中心按面积加权平均。
  - 法线：面片法线按面积加权平均后归一化。
  - 颜色（SH 的 DC 项）：体素内面片颜色均值。
  - 协方差：体素内面片顶点做加权 PCA 得到。
- 不同体素尺寸对应不同 LOD。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python src/mesh_to_3dgs_lod.py --mesh path/to/model.obj --voxel-sizes 0.01,0.02,0.04 --out-dir output_lods
```
如果纹理只有贴图，需要添加参数将纹理bake进mesh
```bash
--bake-textures
```

输出示例：

- `output_lods/lod_00_vox0.0100.ply`
- `output_lods/lod_01_vox0.0200.ply`
- `output_lods/lod_02_vox0.0400.ply`

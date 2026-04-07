# 导出 PLY 参数表

- 文件名格式：`{ply_prefix}_level_{l:02d}_vox{voxel_size_l:.6f}.ply`

---

| 项 | 说明 |
|---|---|
| Header | `ply`, `format binary_little_endian 1.0`, `element vertex N`, `property ...`, `end_header` |
| 数据类型 | 所有属性均为 `float32`（PLY `property float`） |
| 记录顺序 | 与属性声明顺序一致；每行一个 Gaussian |


| 字段名 | 类型 | 含义 | 计算/来源 | 范围 |
|---|---|---|---|---|
| `x`, `y`, `z` | float32 | 高斯中心（世界坐标） | 面片中心按面积加权平均 | 与模型坐标范围一致 |
| `nx`, `ny`, `nz` | float32 | 着色法线方向 | 积分法线面积加权后单位化 | 约 `[-1, 1]` |
| `base_r`, `base_g`, `base_b` | float32 | 基础反照率（sRGB） | 面片 basecolor 积分并面积加权，再夹到 `[0,1]` | `[0,1]` |
| `metallic` | float32 | 金属度 | 面片 metallic 面积加权平均并夹到 `[0,1]` | `[0,1]` |
| `roughness` | float32 | 粗糙度 | 面片 roughness 面积加权平均并夹到 `[0,1]` | `[0,1]` |
| `opacity` | float32 | 不透明度 | 当前固定为 `1.0` | `1.0` |
| `scale_0`, `scale_1`, `scale_2` | float32 | 高斯主轴对数尺度 | `cov -> eigvals -> sqrt -> log` | 取决于几何尺度 |
| `rot_0`, `rot_1`, `rot_2`, `rot_3` | float32 | 旋转四元数 `(w,x,y,z)` | 协方差特征向量转四元数并归一化 | 约 `[-1,1]`，模长约 `1` |

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(gaussians)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float base_r",
        "property float base_g",
        "property float base_b",
        "property float metallic",
        "property float roughness",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n",
    ]
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("base_r", "<f4"),
            ("base_g", "<f4"),
            ("base_b", "<f4"),
            ("metallic", "<f4"),
            ("roughness", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
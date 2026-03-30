from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import trimesh
from tqdm import tqdm
from trimesh.visual import color as visual_color
from trimesh.visual.material import PBRMaterial

EPS = 1e-8


def _barycentric_weights(n_samples: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)
    sr = np.sqrt(r1)
    l1 = 1.0 - sr
    l2 = sr * (1.0 - r2)
    l3 = sr * r2
    return l1, l2, l3


def _integrate_image_per_face(
    image,
    uv_all: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    *,
    seed: int = 0,
    max_uv_per_batch: int = 262_144,
    desc: str = "PBR 采样",
    show_progress: bool = True,
) -> Optional[np.ndarray]:
    """Per-face mean of bilinear-sampled image channels at random UV points (F, C). C = sampled channels."""
    if n_samples <= 0 or image is None:
        return None
    faces = np.asarray(faces, dtype=np.int64)
    uv_tri = uv_all[faces].astype(np.float64, copy=False)
    f_count = uv_tri.shape[0]
    if f_count == 0:
        return None

    rng = np.random.default_rng(seed)
    l1, l2, l3 = _barycentric_weights(n_samples, rng)
    face_batch = max(1, max_uv_per_batch // n_samples)
    # Use RGBA float 0-1 for accumulation
    img_rgba = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
    h, w = img_rgba.shape[0], img_rgba.shape[1]
    mean_ch = np.zeros((f_count, 4), dtype=np.float64)

    for start in tqdm(
        range(0, f_count, face_batch),
        desc=desc,
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
        rgba = _bilinear_rgba(flat, img_rgba, w, h)
        mean_ch[start:end] = rgba.reshape(b, n_samples, 4).mean(axis=1)

    return mean_ch

# 双线性插值 
# 当采样点uv坐标不在像素中心时，使用周围四个像素做插值
def _bilinear_rgba(uv: np.ndarray, img_rgba: np.ndarray, w: int, h: int) -> np.ndarray:
    """uv (N,2) in [0,1]-style; same convention as trimesh (v flipped). Returns (N,4)."""
    x = uv[:, 0] * (w - 1)
    y = (1.0 - uv[:, 1]) * (h - 1)
    x0 = np.floor(x).astype(np.int64) % w
    y0 = np.floor(y).astype(np.int64) % h
    x1 = np.ceil(x).astype(np.int64) % w
    y1 = np.ceil(y).astype(np.int64) % h
    dx = (x % w) - x0
    dy = (y % h) - y0
    dx = np.clip(dx, 0.0, 1.0)
    dy = np.clip(dy, 0.0, 1.0)
    c00 = img_rgba[y0, x0]
    c01 = img_rgba[y0, x1]
    c10 = img_rgba[y1, x0]
    c11 = img_rgba[y1, x1]
    a00 = (1 - dx) * (1 - dy)
    a01 = dx * (1 - dy)
    a10 = (1 - dx) * dy
    a11 = dx * dy
    return (
        c00 * a00[:, None]
        + c01 * a01[:, None]
        + c10 * a10[:, None]
        + c11 * a11[:, None]
    )


def face_constant_tbn(
    verts_tri: np.ndarray, uv_tri: np.ndarray, geom_n: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One orthonormal TBN per face (glTF-style tangent + bitangent from UV)."""
    q0, q1, q2 = verts_tri[0], verts_tri[1], verts_tri[2]
    uv0, uv1, uv2 = uv_tri[0], uv_tri[1], uv_tri[2]
    e1, e2 = q1 - q0, q2 - q0
    duv1, duv2 = uv1 - uv0, uv2 - uv0
    denom = duv1[0] * duv2[1] - duv1[1] * duv2[0]
    if abs(denom) < 1e-12:
        return None
    f = 1.0 / denom
    t = (e1 * duv2[1] - e2 * duv1[1]) * f
    n = geom_n / (np.linalg.norm(geom_n) + EPS)
    t = t - n * np.dot(n, t)
    tn = np.linalg.norm(t)
    if tn < EPS:
        return None
    t = t / tn
    b = np.cross(n, t)
    bn = np.linalg.norm(b)
    if bn < EPS:
        return None
    b = b / bn
    return t, b, n


def decode_normal_map_rgb(rgb: np.ndarray) -> np.ndarray:
    """glTF-style tangent normal from RGB (0–1 float). Output (N,3) unit vectors."""
    x = rgb[:, 0] * 2.0 - 1.0
    y = rgb[:, 1] * 2.0 - 1.0
    z = rgb[:, 2] * 2.0 - 1.0
    v = np.stack([x, y, z], axis=1)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS)

# basecolor积分近似
def pbr_face_basecolor_srgb_integrated(
    mesh: trimesh.Trimesh,
    faces: np.ndarray,
    n_samples: int,
    *,
    seed: int = 0,
    max_uv_per_batch: int = 262_144,
    show_progress: bool = True,
) -> np.ndarray:
    """Per-face base color in sRGB 0–1 (texture texels treated as sRGB)."""
    mat = mesh.visual.material if mesh.visual else None
    uv = getattr(mesh.visual, "uv", None) if mesh.visual else None
    if uv is None or len(uv) != len(mesh.vertices):
        raise ValueError("BRDF path requires per-vertex UV coordinates.")

    if isinstance(mat, PBRMaterial):
        if mat.baseColorTexture is not None:
            ch = _integrate_image_per_face(
                mat.baseColorTexture,
                uv,
                faces,
                n_samples,
                seed=seed,
                max_uv_per_batch=max_uv_per_batch,
                desc="BaseColor 积分",
                show_progress=show_progress,
            )
            if ch is None:
                raise ValueError("Failed to sample baseColorTexture.")
            rgb = ch[:, :3]
            rgb = np.clip(rgb, 0.0, 1.0)
            return rgb
        if mat.baseColorFactor is not None:
            lin = visual_color.to_float(mat.baseColorFactor).reshape(4)[:3]
            srgb = visual_color.linear_to_srgb(np.clip(lin, 0.0, 1.0))
            rgb = np.clip(srgb, 0.0, 1.0)
            return np.tile(rgb.astype(np.float64), (len(faces), 1))

    raise ValueError("BRDF path expects a PBRMaterial with baseColorTexture or baseColorFactor.")

# 着色法线积分近似
def pbr_face_shading_normal_integrated(
    mesh: trimesh.Trimesh,
    faces: np.ndarray,
    geom_normals: np.ndarray,
    n_samples: int,
    *,
    seed: int = 1,
    max_uv_per_batch: int = 262_144,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Per-face world-space shading normal: normal map integrated on triangle (constant TBN per face).
    Falls back to geometric normal if no normal texture or degenerate UV/TBN.
    """
    mat = mesh.visual.material if mesh.visual else None
    uv = getattr(mesh.visual, "uv", None) if mesh.visual else None
    faces = np.asarray(faces, dtype=np.int64)
    f_count = faces.shape[0]
    out = np.array(geom_normals, dtype=np.float64, copy=True)

    if not isinstance(mat, PBRMaterial) or mat.normalTexture is None or uv is None:
        return out

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tri_v = verts[faces]
    uv_tri_full = uv[faces].astype(np.float64, copy=False)

    img = mat.normalTexture.convert("RGB")
    img_rgb = np.asarray(img, dtype=np.float64) / 255.0
    h, w = img_rgb.shape[0], img_rgb.shape[1]
    img_rgba = np.concatenate([img_rgb, np.ones((h, w, 1))], axis=-1)

    rng = np.random.default_rng(seed)
    l1, l2, l3 = _barycentric_weights(n_samples, rng)
    face_batch = max(1, max_uv_per_batch // n_samples)

    for start in tqdm(
        range(0, f_count, face_batch),
        desc="Normal 积分",
        unit="批",
        disable=not show_progress,
    ):
        end = min(start + face_batch, f_count)
        B = end - start
        uv_samples = np.zeros((B, n_samples, 2), dtype=np.float64)
        M_b = np.zeros((B, 3, 3), dtype=np.float64)
        valid = np.zeros(B, dtype=bool)
        for bi in range(B):
            fi = start + bi
            tbn = face_constant_tbn(tri_v[fi], uv_tri_full[fi], geom_normals[fi])
            if tbn is None:
                continue
            t, b, n = tbn
            M_b[bi] = np.column_stack([t, b, n])
            valid[bi] = True
            uvt = uv_tri_full[fi]
            uv_samples[bi] = (
                l1[:, None] * uvt[0]
                + l2[:, None] * uvt[1]
                + l3[:, None] * uvt[2]
            )
        flat = uv_samples.reshape(-1, 2)
        rgba = _bilinear_rgba(flat, img_rgba, w, h)
        nt_b = decode_normal_map_rgb(rgba[:, :3]).reshape(B, n_samples, 3)
        nw = np.einsum("bij,bsj->bsi", M_b, nt_b)
        mean_n = nw.mean(axis=1)
        for bi, fi in enumerate(range(start, end)):
            if not valid[bi]:
                continue
            nn = np.linalg.norm(mean_n[bi])
            if nn > EPS:
                out[fi] = mean_n[bi] / nn

    return out

# 金属度和粗糙度
def pbr_face_metallic_roughness_direct(
    mesh: trimesh.Trimesh,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-face metallic & roughness: centroid UV sample. glTF: G=roughness, B=metallic."""
    mat = mesh.visual.material if mesh.visual else None
    faces = np.asarray(faces, dtype=np.int64)
    f_count = faces.shape[0]
    default_m = 0.0
    default_r = 1.0
    if isinstance(mat, PBRMaterial):
        mf = mat.metallicFactor
        rf = mat.roughnessFactor
        if mf is not None:
            default_m = float(np.asarray(mf).reshape(-1)[0])
        if rf is not None:
            default_r = float(np.asarray(rf).reshape(-1)[0])
    m_out = np.full(f_count, default_m, dtype=np.float64)
    r_out = np.full(f_count, default_r, dtype=np.float64)

    if not isinstance(mat, PBRMaterial) or mat.metallicRoughnessTexture is None:
        return m_out, r_out

    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        return m_out, r_out

    uv_tri = uv[faces].astype(np.float64, copy=False)
    uv_c = uv_tri.mean(axis=1)
    img = mat.metallicRoughnessTexture.convert("RGB")
    img_rgb = np.asarray(img, dtype=np.float64) / 255.0
    h, w = img_rgb.shape[0], img_rgb.shape[1]
    rgba = _bilinear_rgba(uv_c, np.concatenate([img_rgb, np.ones((h, w, 1))], axis=-1), w, h)
    # trimesh / glTF packed: G = roughness, B = metallic (see material.py metallic_roughness fallback)
    r_out[:] = np.clip(rgba[:, 1], 0.0, 1.0)
    m_out[:] = np.clip(rgba[:, 2], 0.0, 1.0)
    print("Metallic,Roughness Finished")
    return m_out, r_out

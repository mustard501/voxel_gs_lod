from __future__ import annotations

from typing import Union

import numpy as np

# 把...abc变成...a00b00c
_PART1BY2_MASK_21 = np.uint64(0x1FFFFF)  # 21 bits
def _part1by2_uint64(n: np.ndarray) -> np.ndarray:
    """
    Expand 21-bit integers so that bits are separated by two zeros.

    Result layout (bit position): out[3*b] = in[b], out[3*b+1] = out[3*b+2] = 0
    """

    n = np.asarray(n, dtype=np.uint64)
    n &= _PART1BY2_MASK_21

    n = (n | (n << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    n = (n | (n << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    n = (n | (n << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    n = (n | (n << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    n = (n | (n << np.uint64(2))) & np.uint64(0x1249249249249249)
    return n

def morton3d_encode(
    x: Union[int, np.ndarray],
    y: Union[int, np.ndarray],
    z: Union[int, np.ndarray],
    *,
    bits_per_axis: int,
) -> np.ndarray:
    """
    Encode 3D integer coordinates (x,y,z) into a 64-bit Morton code.

    Assumes coordinates fit into `bits_per_axis`, and supports up to 21 bits.
    """
    if bits_per_axis > 21:
        raise ValueError(
            f"bits_per_axis={bits_per_axis} too large for uint64 morton encoder; "
            "please reduce scene voxel density or implement 128-bit morton."
        )

    # Vectorized interleave:
    # morton = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
    x_u = np.asarray(x, dtype=np.uint64)
    y_u = np.asarray(y, dtype=np.uint64)
    z_u = np.asarray(z, dtype=np.uint64)

    mx = _part1by2_uint64(x_u)
    my = _part1by2_uint64(y_u) << np.uint64(1)
    mz = _part1by2_uint64(z_u) << np.uint64(2)
    return (mx | my | mz).astype(np.uint64)


def morton3d_parent_key(key: int) -> int:
    """Parent morton key when going one level coarser (shift by 3 bits)."""

    return int(key) >> 3


def morton3d_child_key(parent_key: int, offset_0_to_7: int) -> int:
    """
    Child morton key when going one level finer.

    offset bit mapping:
      dx = offset & 1
      dy = (offset >> 1) & 1
      dz = (offset >> 2) & 1
    """

    return (int(parent_key) << 3) | int(offset_0_to_7)


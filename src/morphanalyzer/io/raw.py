"""Volumes RAW (en-tete absent) — le format d'echange d'iMorph."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["read_raw", "write_raw"]


def read_raw(
    path: str | Path,
    shape: tuple[int, int, int],
    dtype: str | np.dtype = "uint8",
    *,
    offset: int = 0,
    order: str = "C",
    voxel_size: float | tuple[float, float, float] = 1.0,
    unit: str = "um",
    mmap: bool = True,
) -> Volume:
    """Lit un RAW en `(nz, ny, nx)`.

    `mmap=True` (defaut) ne charge rien en memoire : indispensable pour les
    volumes de 2048^3 que produit la ligne ID19.
    """
    path = Path(path)
    dt = np.dtype(dtype)
    expected = int(np.prod(shape)) * dt.itemsize + offset
    actual = path.stat().st_size
    if actual < expected:
        raise ValueError(
            f"{path.name} fait {actual} octets, il en faut {expected} "
            f"pour {shape} en {dt.name} (offset={offset}) — verifier shape/dtype"
        )
    if mmap:
        data = np.memmap(path, dtype=dt, mode="r", offset=offset, shape=shape, order=order)
    else:
        buf = np.fromfile(path, dtype=dt, count=int(np.prod(shape)), offset=offset)
        data = buf.reshape(shape, order=order)
    return Volume(data, voxel_size=voxel_size, unit=unit, name=path.stem)


def write_raw(volume, path: str | Path, *, dtype: str | np.dtype | None = None) -> Path:
    """Ecrit un volume en RAW brut, et un `.json` de description a cote."""
    import json

    path = Path(path)
    arr = as_array(volume)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    arr.tofile(path)
    meta = {"shape": list(arr.shape), "dtype": arr.dtype.name, "order": "C"}
    if isinstance(volume, Volume):
        meta |= {"voxel_size": list(volume.voxel_size), "unit": volume.unit, "name": volume.name}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, indent=2))
    return path

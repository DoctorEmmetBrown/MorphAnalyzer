"""Piles d'images 2D (TIFF multipage ou repertoire d'images numerotees)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["read_stack", "write_stack"]

_EXT = (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")


def read_stack(
    path: str | Path,
    *,
    pattern: str = "*",
    voxel_size: float | tuple[float, float, float] = 1.0,
    unit: str = "um",
    slices: slice | None = None,
) -> Volume:
    """Lit un TIFF multipage, ou un repertoire d'images 2D triees par nom.

    Le tri est *naturel* (image2 avant image10), ce qui evite le piege classique
    du tri lexicographique sur des noms non zero-paddes.
    """
    path = Path(path)
    if path.is_file():
        data = tifffile.imread(path)
        if data.ndim != 3:
            raise ValueError(f"{path.name} contient un tableau {data.ndim}D, 3D attendu")
        if slices is not None:
            data = data[slices]
        return Volume(data, voxel_size=voxel_size, unit=unit, name=path.stem)

    files = sorted(
        (p for p in path.glob(pattern) if p.suffix.lower() in _EXT),
        key=lambda p: _natural_key(p.name),
    )
    if not files:
        raise FileNotFoundError(f"aucune image {_EXT} dans {path} (motif {pattern!r})")
    if slices is not None:
        files = files[slices]
    first = (
        tifffile.imread(files[0])
        if files[0].suffix.lower().startswith(".tif")
        else _imread(files[0])
    )
    out = np.empty((len(files), *first.shape), dtype=first.dtype)
    out[0] = first
    for i, f in enumerate(files[1:], start=1):
        img = tifffile.imread(f) if f.suffix.lower().startswith(".tif") else _imread(f)
        if img.shape != first.shape:
            raise ValueError(f"{f.name} fait {img.shape}, attendu {first.shape}")
        out[i] = img
    return Volume(out, voxel_size=voxel_size, unit=unit, name=path.name)


def _imread(p: Path) -> np.ndarray:
    from skimage.io import imread

    img = imread(p)
    if img.ndim == 3:  # RGB -> gris
        img = img[..., 0]
    return img


def _natural_key(s: str):
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def write_stack(volume, path: str | Path, *, per_slice: bool = False) -> Path:
    """Ecrit un TIFF multipage (defaut) ou un repertoire de coupes."""
    path = Path(path)
    arr = as_array(volume)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    if per_slice:
        path.mkdir(parents=True, exist_ok=True)
        width = len(str(arr.shape[0] - 1))
        for k in range(arr.shape[0]):
            tifffile.imwrite(path / f"slice_{k:0{width}d}.tif", arr[k])
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {}
    if isinstance(volume, Volume):
        dz, dy, dx = volume.voxel_size
        kw = {"resolution": (1.0 / dx, 1.0 / dy), "metadata": {"spacing": dz, "unit": volume.unit}}
    tifffile.imwrite(path, arr, **kw)
    return path

"""Voisinages discrets et elements structurants.

iMorph melangeait 6-, 18- et 26-connexite selon les algorithmes (fast marching
en 6, watershed en 26, composantes connexes en 26). On les nomme explicitement
ici pour que chaque portage documente son choix.
"""

from __future__ import annotations

import functools

import numpy as np

__all__ = ["connectivity_offsets", "ball_offsets", "ball_structure"]

_CONNECTIVITY = {6: 1, 18: 2, 26: 3}


@functools.lru_cache(maxsize=8)
def connectivity_offsets(connectivity: int = 26) -> np.ndarray:
    """Decalages `(n, 3)` des voisins pour 6-, 18- ou 26-connexite.

    Le voxel central `(0, 0, 0)` est exclu.
    """
    if connectivity not in _CONNECTIVITY:
        raise ValueError(f"connexite attendue parmi {sorted(_CONNECTIVITY)}, recu {connectivity}")
    rank = _CONNECTIVITY[connectivity]
    off = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0) and (abs(dz) + abs(dy) + abs(dx)) <= rank
    ]
    return np.array(off, dtype=np.int64)


@functools.lru_cache(maxsize=64)
def ball_offsets(radius: float) -> np.ndarray:
    """Decalages `(n, 3)` des voxels d'une boule discrete de rayon `radius`.

    Critere `floor(d) <= radius`, celui d'iMorph (`morphology.cpp`, construction
    de `BouleMask`), pour que la granulometrie portee soit comparable voxel a
    voxel avec la version C++.
    """
    r = int(np.ceil(radius))
    g = np.arange(-r, r + 1)
    dz, dy, dx = np.meshgrid(g, g, g, indexing="ij")
    d = np.sqrt(dz * dz + dy * dy + dx * dx)
    keep = np.floor(d) <= radius
    return np.stack([dz[keep], dy[keep], dx[keep]], axis=1).astype(np.int64)


@functools.lru_cache(maxsize=64)
def ball_structure(radius: float) -> np.ndarray:
    """Element structurant booleen spherique, meme critere que `ball_offsets`."""
    r = int(np.ceil(radius))
    g = np.arange(-r, r + 1)
    dz, dy, dx = np.meshgrid(g, g, g, indexing="ij")
    return np.floor(np.sqrt(dz * dz + dy * dy + dx * dx)) <= radius

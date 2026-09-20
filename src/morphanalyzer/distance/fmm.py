"""Fast marching : resolution de l'equation eikonale.

Portage de `Thread/Granulometry/fastMarchManu.cpp` (schema du 1er ordre,
`computeDistanceFirstOrder`, `computeDistanceFirstOrderWithVelocityField`).

L'equation eikonale ``|grad T| F = 1`` decrit le temps d'arrivee `T` d'un front
qui avance a la vitesse `F` dans la direction de sa normale. Trois usages dans
cette bibliotheque :

- **geodesiques** : `F = 1` confine a une phase, donc `T` est la distance
  geodesique, celle qui contourne les obstacles ;
- **propagation de Poiseuille** : `F` varie selon un profil parabolique, ce qui
  donne des chemins centres dans les constrictions (these §3.2.3) ;
- tortuosite, comme rapport de `T` a la distance euclidienne.

!!! note "Pas pour la distance euclidienne"
    Pour `F = 1` sans obstacle, `distance.distance_transform` est **exacte** et
    plus rapide. La these mesure jusqu'a 2,77 voxels d'erreur pour le fast
    marching du 1er ordre (fig. 3.19). Le fast marching sert quand la distance
    euclidienne ne repond pas a la question : contournement, vitesse variable.
"""

from __future__ import annotations

import numpy as np

from morphanalyzer._deps import have
from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["travel_time", "geodesic_distance"]

_FAR, _BAND, _FROZEN = 0, 1, 2


def _kernel():
    from numba import njit

    @njit(cache=True, nogil=True)
    def _run(speed, valid, sources, spacing, out):  # pragma: no cover - numba
        nz, ny, nx = speed.shape
        state = np.empty(speed.shape, dtype=np.uint8)
        big = np.float32(1.0e30)
        n_valid = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    out[k, j, i] = big
                    if valid[k, j, i]:
                        state[k, j, i] = _FAR
                        n_valid += 1
                    else:
                        state[k, j, i] = _FROZEN

        w = np.empty(3, dtype=np.float64)
        for a in range(3):
            w[a] = 1.0 / (spacing[a] * spacing[a])

        cap = n_valid + 1
        hkey = np.empty(cap, dtype=np.float64)
        hidx = np.empty(cap, dtype=np.int64)
        size = 0

        # les sources sont gelees a T = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if sources[k, j, i] and valid[k, j, i]:
                        out[k, j, i] = 0.0
                        state[k, j, i] = _FROZEN

        vals = np.empty(3, dtype=np.float64)
        wsel = np.empty(3, dtype=np.float64)

        # amorce la bande etroite avec les voisins des sources
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if not (sources[k, j, i] and valid[k, j, i]):
                        continue
                    for a in range(3):
                        for d in (-1, 1):
                            kk = k + d if a == 0 else k
                            jj = j + d if a == 1 else j
                            ii = i + d if a == 2 else i
                            if kk < 0 or kk >= nz or jj < 0 or jj >= ny or ii < 0 or ii >= nx:
                                continue
                            if state[kk, jj, ii] == _FROZEN:
                                continue
                            t = _local(out, state, speed, w, vals, wsel, kk, jj, ii, nz, ny, nx)
                            if t < out[kk, jj, ii]:
                                out[kk, jj, ii] = np.float32(t)
                            if state[kk, jj, ii] == _FAR:
                                state[kk, jj, ii] = _BAND
                            p = size
                            size += 1
                            hkey[p] = out[kk, jj, ii]
                            hidx[p] = (kk * ny + jj) * nx + ii
                            while p > 0:
                                q = (p - 1) // 2
                                if hkey[q] <= hkey[p]:
                                    break
                                hkey[p], hkey[q] = hkey[q], hkey[p]
                                hidx[p], hidx[q] = hidx[q], hidx[p]
                                p = q

        while size > 0:
            key = hkey[0]
            flat = hidx[0]
            size -= 1
            hkey[0] = hkey[size]
            hidx[0] = hidx[size]
            p = 0
            while True:
                left = 2 * p + 1
                if left >= size:
                    break
                small = left
                right = left + 1
                if right < size and hkey[right] < hkey[left]:
                    small = right
                if hkey[p] <= hkey[small]:
                    break
                hkey[p], hkey[small] = hkey[small], hkey[p]
                hidx[p], hidx[small] = hidx[small], hidx[p]
                p = small

            k = flat // (ny * nx)
            rem = flat - k * ny * nx
            j = rem // nx
            i = rem - j * nx
            if state[k, j, i] == _FROZEN or key > out[k, j, i]:
                continue  # entree perimee dans le tas
            state[k, j, i] = _FROZEN

            for a in range(3):
                for d in (-1, 1):
                    kk = k + d if a == 0 else k
                    jj = j + d if a == 1 else j
                    ii = i + d if a == 2 else i
                    if kk < 0 or kk >= nz or jj < 0 or jj >= ny or ii < 0 or ii >= nx:
                        continue
                    if state[kk, jj, ii] == _FROZEN:
                        continue
                    t = _local(out, state, speed, w, vals, wsel, kk, jj, ii, nz, ny, nx)
                    if t < out[kk, jj, ii]:
                        out[kk, jj, ii] = np.float32(t)
                        state[kk, jj, ii] = _BAND
                        p = size
                        size += 1
                        # la cle doit etre la valeur *stockee*, arrondie en
                        # float32 : comparer plus tard une cle float64 a une
                        # carte float32 fait rejeter des entrees valides comme
                        # perimees, et les distances sortent trop grandes
                        hkey[p] = out[kk, jj, ii]
                        hidx[p] = (kk * ny + jj) * nx + ii
                        while p > 0:
                            q = (p - 1) // 2
                            if hkey[q] <= hkey[p]:
                                break
                            hkey[p], hkey[q] = hkey[q], hkey[p]
                            hidx[p], hidx[q] = hidx[q], hidx[p]
                            p = q
        return out

    @njit(cache=True, nogil=True, inline="always")
    def _local(out, state, speed, w, vals, wsel, k, j, i, nz, ny, nx):  # pragma: no cover
        """Schema UPWIND du 1er ordre en (k, j, i), d'apres les voisins geles."""
        n = 0
        for a in range(3):
            best = 1.0e30
            for d in (-1, 1):
                kk = k + d if a == 0 else k
                jj = j + d if a == 1 else j
                ii = i + d if a == 2 else i
                if kk < 0 or kk >= nz or jj < 0 or jj >= ny or ii < 0 or ii >= nx:
                    continue
                if state[kk, jj, ii] == _FROZEN and out[kk, jj, ii] < best:
                    best = out[kk, jj, ii]
            if best < 1.0e29:
                vals[n] = best
                wsel[n] = w[a]
                n += 1
        if n == 0:
            return 1.0e30

        # tri croissant des au plus trois valeurs
        for a in range(n):
            for b in range(a + 1, n):
                if vals[b] < vals[a]:
                    vals[a], vals[b] = vals[b], vals[a]
                    wsel[a], wsel[b] = wsel[b], wsel[a]

        f = speed[k, j, i]
        if f <= 0.0:
            return 1.0e30
        rhs = 1.0 / (f * f)

        # on essaie d'utiliser les 3, puis 2, puis 1 direction : la solution
        # retenue est la premiere qui reste au-dessus de toutes les valeurs
        # amont employees (condition de causalite du schema upwind)
        for m in range(n, 0, -1):
            A = 0.0
            B = 0.0
            C = -rhs
            for a in range(m):
                A += wsel[a]
                B -= 2.0 * wsel[a] * vals[a]
                C += wsel[a] * vals[a] * vals[a]
            disc = B * B - 4.0 * A * C
            if disc < 0.0:
                continue
            t = (-B + np.sqrt(disc)) / (2.0 * A)
            if t >= vals[m - 1]:
                return t
        return vals[0] + 1.0 / (f * np.sqrt(wsel[0]))

    return _run


_RUN = None


def travel_time(
    mask,
    sources,
    *,
    speed=None,
    cost=None,
    voxel_size=None,
    method: str = "auto",
):
    """Temps d'arrivee d'un front partant de `sources`, confine a `mask`.

    Parameters
    ----------
    mask
        Domaine de propagation (la phase consideree).
    sources
        Masque booleen des points de depart, ou un triplet de coordonnees, ou
        un entier `0..5` designant une face du volume (`0` = z minimal, `1` = z
        maximal, `2` = y minimal, etc.) — la « propagation depuis un plan » de
        la these.
    speed
        Champ de vitesse `F > 0`, meme forme que `mask`. `None` vaut 1 partout,
        donc `T` est la distance geodesique. Convention standard :
        `|grad T| = 1 / F`, donc un `F` grand fait avancer le front vite.
    cost
        Lenteur `1 / F`, en alternative a `speed`. Exclusif de `speed`.

        Cette option existe parce qu'iMorph resout `|grad T| = F` et non
        `|grad T| F = 1` (`computeDistanceFirstOrderWithVelocityField` pose
        `fieldValue = F * F` au second membre) : ce que la these appelle un
        « champ de vitesse » est donc une **lenteur**. C'est ce qui reconcilie
        l'equation 3.11 avec ses propres figures — sans quoi les chemins
        longeraient les parois au lieu d'etre centres.
    method
        `"auto"` : noyau Numba si disponible, sinon `scikit-fmm`. Aucun des deux
        n'est dans le noyau : le fast marching demande l'extra `fast` ou `fmm`.

    Returns
    -------
    numpy.ndarray
        `float32`, `+inf` hors du masque et sur les voxels non atteints.
    """
    global _RUN
    m = as_array(mask).astype(bool, copy=False)
    if voxel_size is None:
        voxel_size = mask.voxel_size if isinstance(mask, Volume) else (1.0, 1.0, 1.0)
    if np.isscalar(voxel_size):
        voxel_size = (float(voxel_size),) * 3
    spacing = np.asarray(voxel_size, dtype=np.float64)

    src = _as_source_mask(sources, m.shape) & m
    if not src.any():
        raise ValueError(
            "aucune source dans le masque : verifier `sources` (une face vide "
            "signifie que la phase ne touche pas cette face)"
        )

    if speed is not None and cost is not None:
        raise ValueError("passer `speed` ou `cost`, pas les deux")
    if cost is not None:
        c = as_array(cost).astype(np.float32, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(c > 0, 1.0 / np.maximum(c, 1e-12), np.float32(1e12))
        f = f.astype(np.float32)
    elif speed is None:
        f = np.ones(m.shape, dtype=np.float32)
    else:
        f = as_array(speed).astype(np.float32, copy=False)

    use = method
    if use == "auto":
        use = "numba" if have("numba") else ("skfmm" if have("skfmm") else "none")
    if use == "numba":
        from morphanalyzer._deps import require

        require("numba", reason="le fast marching")
        if _RUN is None:
            _RUN = _kernel()
        out = np.empty(m.shape, dtype=np.float32)
        _RUN(np.ascontiguousarray(f), m, src, spacing, out)
        out[out > 1.0e29] = np.inf
    elif use == "skfmm":
        import skfmm

        phi = np.ma.MaskedArray(np.where(src, -1.0, 1.0), mask=~m)
        out = np.asarray(skfmm.travel_time(phi, speed=f, dx=tuple(spacing)), dtype=np.float32)
        out = np.where(m, np.ma.filled(out, np.inf), np.inf).astype(np.float32)
    else:
        raise ImportError(
            "le fast marching demande numba ou scikit-fmm. Installer avec :\n"
            '  pip install "morphanalyzer[fast]"     # noyau Numba, fidele a iMorph\n'
            '  pip install "morphanalyzer[fmm]"      # scikit-fmm'
        )
    return mask.with_data(out, name="travel_time") if isinstance(mask, Volume) else out


def geodesic_distance(mask, sources, *, voxel_size=None, method: str = "auto"):
    """Distance geodesique dans `mask` : `travel_time` a vitesse unite."""
    return travel_time(mask, sources, speed=None, voxel_size=voxel_size, method=method)


def _as_source_mask(sources, shape) -> np.ndarray:
    """Accepte un masque, un point, une liste de points, ou un numero de face."""
    if isinstance(sources, (int, np.integer)):
        face = int(sources)
        if not 0 <= face <= 5:
            raise ValueError("un numero de face doit etre dans 0..5")
        out = np.zeros(shape, dtype=bool)
        axis, side = face // 2, face % 2
        sl = [slice(None)] * 3
        sl[axis] = slice(-1, None) if side else slice(0, 1)
        out[tuple(sl)] = True
        return out
    arr = np.asarray(sources)
    if arr.dtype == bool and arr.shape == tuple(shape):
        return arr
    if arr.ndim == 1 and arr.size == 3:
        out = np.zeros(shape, dtype=bool)
        out[tuple(int(v) for v in arr)] = True
        return out
    if arr.ndim == 2 and arr.shape[1] == 3:
        out = np.zeros(shape, dtype=bool)
        out[tuple(arr.astype(int).T)] = True
        return out
    raise ValueError(
        "sources doit etre un masque booleen, un point (k, j, i), un tableau "
        "(n, 3) de points, ou un numero de face 0..5"
    )

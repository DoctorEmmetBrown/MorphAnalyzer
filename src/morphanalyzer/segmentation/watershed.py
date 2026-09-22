"""Ligne de partage des eaux a priorites reelles.

Portage de `Thread/Granulometry/morphology.cpp::watershedBinarySearchTree` et
`mostRepresenatedlabel3`.

Pourquoi ne pas simplement appeler `skimage.segmentation.watershed`. Celle-ci
implemente l'inondation de Meyer sur file d'attente hierarchique, qui **discretise
le relief** en niveaux entiers. Sur une carte de distance a valeurs reelles, cette
quantification rend le resultat dependant de l'ordre d'insertion sur les plateaux,
et produit les lignes de partage en « marches d'escalier » que la these documente
(fig. 3.4b). iMorph a remplace la file hierarchique par un **tas binaire**, qui
accepte des priorites dans R, et resolu les collisions par **le label le plus
represente** dans le voisinage deja traite — a egalite, le plus petit label. C'est
cette variante qui est portee ici (fig. 3.5).
"""

from __future__ import annotations

import warnings

import numpy as np

from morphanalyzer._deps import have
from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["watershed"]

_NOT_YET, _IN_HEAP, _DEALED = 0, 1, 2


def _kernel_source():
    """Le noyau numba, compile a la demande (l'import reste leger sans numba)."""
    from numba import njit

    @njit(cache=True, nogil=True)
    def _run(relief, valid, markers, out):  # pragma: no cover - compile par numba
        nz, ny, nx = relief.shape
        state = np.empty(relief.shape, dtype=np.uint8)
        rmax = np.float32(-1.0e30)
        n_valid = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    out[k, j, i] = -1
                    if valid[k, j, i]:
                        state[k, j, i] = _NOT_YET
                        n_valid += 1
                        if relief[k, j, i] > rmax:
                            rmax = relief[k, j, i]
                    else:
                        state[k, j, i] = _DEALED

        cap = n_valid + 1
        hkey = np.empty(cap, dtype=np.float32)
        hidx = np.empty(cap, dtype=np.int64)
        size = 0

        # --- tas binaire minimal sur (cle, indice aplati) -------------------
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if markers[k, j, i] > 0:
                        state[k, j, i] = _DEALED
                        out[k, j, i] = markers[k, j, i]

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if markers[k, j, i] <= 0:
                        continue
                    for dk in range(-1, 2):
                        kk = k + dk
                        if kk < 0 or kk >= nz:
                            continue
                        for dj in range(-1, 2):
                            jj = j + dj
                            if jj < 0 or jj >= ny:
                                continue
                            for di in range(-1, 2):
                                ii = i + di
                                if ii < 0 or ii >= nx:
                                    continue
                                if state[kk, jj, ii] != _NOT_YET:
                                    continue
                                state[kk, jj, ii] = _IN_HEAP
                                key = rmax - relief[kk, jj, ii]
                                p = size
                                size += 1
                                hkey[p] = key
                                hidx[p] = (kk * ny + jj) * nx + ii
                                while p > 0:
                                    q = (p - 1) // 2
                                    if hkey[q] <= hkey[p]:
                                        break
                                    hkey[p], hkey[q] = hkey[q], hkey[p]
                                    hidx[p], hidx[q] = hidx[q], hidx[p]
                                    p = q

        tab = np.empty(27, dtype=np.int64)
        vals = np.empty(27, dtype=np.int64)
        cnts = np.empty(27, dtype=np.int64)

        while size > 0:
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

            if state[k, j, i] != _DEALED:
                state[k, j, i] = _DEALED
                n = 0
                for dk in range(-1, 2):
                    kk = k + dk
                    if kk < 0 or kk >= nz:
                        continue
                    for dj in range(-1, 2):
                        jj = j + dj
                        if jj < 0 or jj >= ny:
                            continue
                        for di in range(-1, 2):
                            ii = i + di
                            if ii < 0 or ii >= nx:
                                continue
                            if state[kk, jj, ii] == _DEALED and out[kk, jj, ii] > -1:
                                tab[n] = out[kk, jj, ii]
                                n += 1
                # label le plus represente, a egalite le plus petit
                ndiff = 0
                for c in range(n):
                    found = False
                    for d in range(ndiff):
                        if vals[d] == tab[c]:
                            cnts[d] += 1
                            found = True
                            break
                    if not found:
                        vals[ndiff] = tab[c]
                        cnts[ndiff] = 1
                        ndiff += 1
                best = -1
                bestc = 0
                for d in range(ndiff):
                    if cnts[d] > bestc or (cnts[d] == bestc and vals[d] < best):
                        bestc = cnts[d]
                        best = vals[d]
                out[k, j, i] = best

            for dk in range(-1, 2):
                kk = k + dk
                if kk < 0 or kk >= nz:
                    continue
                for dj in range(-1, 2):
                    jj = j + dj
                    if jj < 0 or jj >= ny:
                        continue
                    for di in range(-1, 2):
                        ii = i + di
                        if ii < 0 or ii >= nx:
                            continue
                        if state[kk, jj, ii] != _NOT_YET:
                            continue
                        state[kk, jj, ii] = _IN_HEAP
                        key = rmax - relief[kk, jj, ii]
                        p = size
                        size += 1
                        hkey[p] = key
                        hidx[p] = (kk * ny + jj) * nx + ii
                        while p > 0:
                            q = (p - 1) // 2
                            if hkey[q] <= hkey[p]:
                                break
                            hkey[p], hkey[q] = hkey[q], hkey[p]
                            hidx[p], hidx[q] = hidx[q], hidx[p]
                            p = q
        return out

    return _run


_RUN = None


def watershed(
    relief,
    markers,
    *,
    mask=None,
    method: str = "auto",
    invert: bool = True,
):
    """Inonde `relief` depuis `markers`, en 26-connexite.

    Parameters
    ----------
    relief
        Relief topographique a valeurs reelles. Typiquement la carte de distance
        a la phase opposee : les centres de cellules en sont les maxima.
    markers
        Image de labels entiers, `0` = pas un marqueur, `1..n` = germes.
    mask
        Domaine valide. Par defaut, la ou `relief > 0`.
    method
        `"auto"` (defaut) : le noyau fidele en Numba si numba est installe,
        sinon `skimage` avec un avertissement. `"numba"` exige Numba.
        `"skimage"` force la variante a relief quantifie.
    invert
        Inonder depuis les **maxima** du relief (defaut), comme l'« immersion
        inversee » de la these : sur une carte de distance, les vallees sont les
        maxima locaux. C'est le bon choix pour segmenter des cellules a partir de
        marqueurs situes en leur centre. Mettre `False` pour inonder depuis les
        minima — le bon choix quand les germes sont sur le **bord** du domaine,
        par exemple pour propager des labels de fluide dans le solide.

        Ce n'est pas un detail de convention. L'algorithme est un parcours
        « meilleur d'abord » sur le relief, pas une immersion synchronisee par
        niveaux : le premier front qui atteint la crete l'emporte ensuite
        largement. Une plaque de 4 voxels entre deux cellules se partage 800/800
        en inondant depuis l'interface, et 1600/0 en inondant depuis la crete.
        Voir `skeleton.plateau_skeleton` pour la mesure de l'effet sur un cas
        ou la reponse exacte est connue.

    Returns
    -------
    numpy.ndarray
        Labels `int32`, `0` hors du masque. Il n'y a **pas** de ligne de partage
        materialisee : chaque voxel du masque recoit un label, et la frontiere
        entre deux cellules est implicite. C'est le choix d'iMorph, et il evite
        d'avoir a decider a quelle cellule appartient un voxel de digue.
    """
    global _RUN
    r = as_array(relief).astype(np.float32, copy=False)
    mk = as_array(markers).astype(np.int32, copy=False)
    if mk.shape != r.shape:
        raise ValueError(f"marqueurs de forme {mk.shape}, relief de forme {r.shape}")
    valid = (r > 0) if mask is None else as_array(mask).astype(bool, copy=False)
    if not (mk > 0).any():
        raise ValueError(
            "aucun marqueur : l'image de marqueurs est vide. Causes usuelles : "
            "`cell_markers` a tourne sur la mauvaise phase (les brins d'une mousse "
            "sont trop fins pour contenir une boule de rayon min_radius), ou "
            "min_radius / fill_ratio sont trop severes."
        )
    # Les marqueurs n'ont pas a etre dans le masque. iMorph s'en sert ainsi pour
    # propager les labels de cellules *dans le solide* : les germes sont les
    # voxels de fluide deja etiquetes, juste a l'exterieur du domaine inonde, et
    # l'inondation part de leurs voisins. Ce qui compte est donc qu'un marqueur
    # touche le masque, pas qu'il y soit.
    from scipy import ndimage as _ndi

    if not (_ndi.binary_dilation(mk > 0, structure=np.ones((3, 3, 3), dtype=bool)) & valid).any():
        n_mk = int((mk > 0).sum())
        n_in = int(((mk > 0) & valid).sum())
        raise ValueError(
            f"aucun marqueur adjacent au masque : l'inondation ne peut pas demarrer. "
            f"{n_mk} marqueur(s), dont {n_in} dans le masque, qui couvre "
            f"{100 * valid.mean():.1f} % du volume. "
            "Le cas le plus courant est un masque pris sur l'autre phase : des "
            "marqueurs places au centre des pores sont loin du solide, et leur "
            "voisinage immediat ne le touche pas. Verifier que `mask` et les "
            "marqueurs designent la meme phase."
        )

    use = method
    if use == "auto":
        use = "numba" if have("numba") else "skimage"
        if use == "skimage":
            warnings.warn(
                "numba absent : repli sur skimage.segmentation.watershed. Ce n'est PAS "
                "l'algorithme de la these : skimage quantifie le relief et ne resout pas "
                "les collisions par label majoritaire, ce qui redonne les frontieres en "
                "marches d'escalier de la figure 3.4 au lieu de la figure 3.5 (mesure sur "
                "une mousse de Voronoi : +15 % de surface d'interface, 9 % des voxels "
                'attribues autrement). Pour la variante fidele : pip install "morphanalyzer[fast]".',
                RuntimeWarning,
                stacklevel=2,
            )

    if use == "numba":
        from morphanalyzer._deps import require

        require("numba", reason="le watershed a priorites reelles")
        if _RUN is None:
            _RUN = _kernel_source()
        work = r if invert else (float(r.max()) - r).astype(np.float32)
        out = np.empty(r.shape, dtype=np.int32)
        _RUN(np.ascontiguousarray(work), valid, mk, out)
        out[~valid] = 0
        out[out < 0] = 0
    elif use == "skimage":
        from skimage.segmentation import watershed as _ws

        work = -r if invert else r
        out = _ws(work, markers=mk, mask=valid, connectivity=3).astype(np.int32)
    else:
        raise ValueError("method doit valoir 'auto', 'numba' ou 'skimage'")

    if isinstance(relief, Volume):
        return relief.with_data(out, name="cells")
    return out

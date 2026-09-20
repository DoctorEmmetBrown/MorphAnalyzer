"""Export du reseau de pores vers OpenPNM et vers le format graphe d'iMorph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from morphanalyzer._deps import require

__all__ = ["to_openpnm", "save_network_text"]


def to_openpnm(
    cells: pd.DataFrame,
    throats: pd.DataFrame,
    *,
    length_scale: float = 1.0,
    cell_radius=None,
    throat_radius=None,
):
    """Construit un `openpnm.network.Network` a partir des deux tables.

    Parameters
    ----------
    length_scale
        Facteur applique aux longueurs pour passer en metres (OpenPNM travaille
        en SI). Pour des coordonnees en micrometres, `1e-6`.
    cell_radius, throat_radius
        Nom de colonne ou tableau. Par defaut `d_equivalent / 2`.

    Returns
    -------
    openpnm.network.Network
        Avec `pore.coords`, `pore.diameter`, `pore.volume`, `throat.conns`,
        `throat.diameter`, `throat.area`.
    """
    op = require("openpnm", reason="l'export vers OpenPNM")
    lab = cells["label"].to_numpy()
    index = {int(v): i for i, v in enumerate(lab)}

    coords = (
        np.stack(
            [
                cells["centroid_x"].to_numpy(dtype=float),
                cells["centroid_y"].to_numpy(dtype=float),
                cells["centroid_z"].to_numpy(dtype=float),
            ],
            axis=1,
        )
        * length_scale
    )

    ta = np.array([index.get(int(v), -1) for v in throats["label_a"]], dtype=int)
    tb = np.array([index.get(int(v), -1) for v in throats["label_b"]], dtype=int)
    ok = (ta >= 0) & (tb >= 0)
    conns = np.stack([ta[ok], tb[ok]], axis=1)

    net = op.network.Network(coords=coords, conns=conns)
    if cell_radius is None:
        pd_ = cells["d_equivalent"].to_numpy(dtype=float)
    elif isinstance(cell_radius, str):
        pd_ = 2.0 * cells[cell_radius].to_numpy(dtype=float)
    else:
        pd_ = 2.0 * np.asarray(cell_radius, dtype=float)
    net["pore.diameter"] = pd_ * length_scale
    if "volume" in cells:
        net["pore.volume"] = cells["volume"].to_numpy(dtype=float) * length_scale**3

    if throat_radius is None:
        td = throats["d_equivalent"].to_numpy(dtype=float)[ok]
    elif isinstance(throat_radius, str):
        td = 2.0 * throats[throat_radius].to_numpy(dtype=float)[ok]
    else:
        td = 2.0 * np.asarray(throat_radius, dtype=float)[ok]
    net["throat.diameter"] = td * length_scale
    if "area" in throats:
        net["throat.area"] = throats["area"].to_numpy(dtype=float)[ok] * length_scale**2
    return net


def save_network_text(
    cells: pd.DataFrame,
    throats: pd.DataFrame,
    path,
    *,
    length_scale: float = 1.0,
) -> Path:
    """Ecrit le reseau au format texte « Geometry File » d'iMorph.

    C'est le format que produisait `invasionIPThread::testingFile`, lisible par
    les outils de reseau de pores de l'IUSTI. Les longueurs sont multipliees
    par `length_scale`.
    """
    path = Path(path)
    lab = cells["label"].to_numpy()
    index = {int(v): i for i, v in enumerate(lab)}
    n_p = len(lab)
    ta = np.array([index.get(int(v), -1) for v in throats["label_a"]], dtype=int)
    tb = np.array([index.get(int(v), -1) for v in throats["label_b"]], dtype=int)
    ok = (ta >= 0) & (tb >= 0)
    ta, tb = ta[ok], tb[ok]
    th = throats[ok].reset_index(drop=True)

    lines = ["Geometry File", "", "Network Position", "0 0 0", "", "Number Of Pores :", str(n_p)]
    lines += ["", "(Id Diam X Y Z Domain)"]
    for i in range(n_p):
        row = cells.iloc[i]
        lines.append(
            f"{i} {float(row['d_equivalent']) * length_scale:g} "
            f"{float(row['centroid_x']) * length_scale:g} "
            f"{float(row['centroid_y']) * length_scale:g} "
            f"{float(row['centroid_z']) * length_scale:g} CLL Cubic"
        )
    lines += [
        "",
        "Number Of Throats :",
        str(len(th)),
        "",
        "(Id Diam X Y Z Domain Label Orientation)",
    ]
    for i in range(len(th)):
        row = th.iloc[i]
        lines.append(
            f"{i} {float(row['d_equivalent']) * length_scale:g} "
            f"{float(row['centroid_x']) * length_scale:g} "
            f"{float(row['centroid_y']) * length_scale:g} "
            f"{float(row['centroid_z']) * length_scale:g} CLL BoundaryInitialized Cubic 2"
        )
    lines += ["", "Pore Linking"]
    nbr: list[list[int]] = [[] for _ in range(n_p)]
    for j, (a, b) in enumerate(zip(ta, tb, strict=True)):
        nbr[a].append(j)
        nbr[b].append(j)
    for i in range(n_p):
        lines.append(f"{i} " + " ".join(str(j) for j in nbr[i]))
    lines += ["", "Throat Linking"]
    for j, (a, b) in enumerate(zip(ta, tb, strict=True)):
        lines.append(f"{j} {a} {b}")
    path.write_text("\n".join(lines) + "\n")
    return path

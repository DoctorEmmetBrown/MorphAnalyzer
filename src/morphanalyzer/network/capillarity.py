"""Loi de Young-Laplace : rayon <-> pression capillaire.

Portage de la conversion faite dans
`Thread/Granulometry/utility.cpp::saveFractionFile` et
`PhysicalModules/PoreNetworkModelling/invasionIPThread.cpp`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "capillary_pressure",
    "capillary_radius",
    "LENGTH_UNITS",
    "WATER_AIR_SURFACE_TENSION",
    "MERCURY_SURFACE_TENSION",
    "MERCURY_CONTACT_ANGLE",
]

#: Facteurs de conversion vers le metre.
LENGTH_UNITS = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "micron": 1e-6,
    "nm": 1e-9,
    "voxel": 1.0,
    "px": 1.0,
}

#: Tension superficielle eau/air a 20 degres C, en N/m.
WATER_AIR_SURFACE_TENSION = 0.0728
#: Tension superficielle du mercure, en N/m.
MERCURY_SURFACE_TENSION = 0.485
#: Angle de contact usuel du mercure, en degres.
MERCURY_CONTACT_ANGLE = 140.0


def _scale(unit: str) -> float:
    try:
        return LENGTH_UNITS[str(unit)]
    except KeyError:
        raise ValueError(
            f"unite de longueur inconnue : {unit!r}. Valeurs acceptees : "
            + ", ".join(sorted(LENGTH_UNITS))
        ) from None


def capillary_pressure(
    radius,
    *,
    surface_tension: float = WATER_AIR_SURFACE_TENSION,
    contact_angle: float = 0.0,
    unit: str = "um",
    convention: str = "laplace",
):
    """Pression capillaire, en pascals, associee a un rayon de courbure.

    Parameters
    ----------
    radius
        Rayon, exprime dans l'unite `unit`. Scalaire ou tableau. Les rayons
        nuls ou negatifs donnent `inf`.
    surface_tension
        Tension superficielle en N/m. Defaut : eau/air a 20 degres C.
    contact_angle
        Angle de contact en **degres**. 0 = mouillage parfait.
    unit
        Unite de `radius`. Defaut `"um"`, l'unite de travail habituelle en
        tomographie et celle que supposait iMorph.
    convention
        `"laplace"` (defaut) applique `Pc = 2 sigma cos(theta) / r`, la loi de
        Young-Laplace pour un menisque spherique de rayon `r`.

        `"imorph"` applique `Pc = 4 sigma cos(theta) / r`, la formule ecrite
        dans iMorph. Elle revient a interpreter la valeur de la carte
        d'ouverture comme un **diametre** et non comme un rayon : iMorph
        ecrivait `4 sigma / d` avec `d` la colonne intitulee « radius ball ».
        Le facteur 2 d'ecart est systematique, donc sans effet sur la forme de
        la courbe de retention, mais il decale l'axe des pressions. On la
        conserve pour pouvoir reproduire les sorties historiques.

    Notes
    -----
    La convention d'iMorph n'appliquait aucun angle de contact : elle
    correspond a `contact_angle=0`.
    """
    if convention not in ("laplace", "imorph"):
        raise ValueError("convention doit valoir 'laplace' ou 'imorph'")
    factor = 2.0 if convention == "laplace" else 4.0
    r = np.asarray(radius, dtype=float) * _scale(unit)
    cos = float(np.cos(np.deg2rad(contact_angle)))
    with np.errstate(divide="ignore", invalid="ignore"):
        pc = factor * surface_tension * cos / r
    pc = np.where(r > 0, pc, np.inf)
    return float(pc) if np.ndim(radius) == 0 else pc


def capillary_radius(
    pressure,
    *,
    surface_tension: float = WATER_AIR_SURFACE_TENSION,
    contact_angle: float = 0.0,
    unit: str = "um",
    convention: str = "laplace",
):
    """Reciproque de :func:`capillary_pressure` : rayon accessible a `pressure`."""
    if convention not in ("laplace", "imorph"):
        raise ValueError("convention doit valoir 'laplace' ou 'imorph'")
    factor = 2.0 if convention == "laplace" else 4.0
    p = np.asarray(pressure, dtype=float)
    cos = float(np.cos(np.deg2rad(contact_angle)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = factor * surface_tension * cos / p / _scale(unit)
    r = np.where(p > 0, r, np.inf)
    return float(r) if np.ndim(pressure) == 0 else r

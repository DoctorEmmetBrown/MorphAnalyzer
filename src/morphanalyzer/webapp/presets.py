"""Chaines types : ce que l'on enchaine presque toujours.

Ces chaines ne sont pas de la decoration d'interface. Elles encodent l'ordre
**correct** des etapes, et cet ordre a un piege : chaque etape recoit par defaut
la sortie de la precedente, alors que plusieurs d'entre elles ont besoin d'un
resultat plus ancien. Le cas qui fait mal :

    distance_transform  -> distance     le volume courant devient la distance
    cell_markers        -> marqueurs    il devient les marqueurs
    watershed_cells                     il lui faut la DISTANCE comme relief

Sans `input`, la derniere etape inonde l'image des marqueurs. Elle rend une
partition qui *ressemble* a des cellules — c'est a peu pres un Voronoi
euclidien des marqueurs — mais l'IoU contre la partition exacte tombe de 0,908 a
0,636. D'ou `input` explicite, et un test qui rejoue chaque chaine et verifie
son resultat contre la verite terrain.

Les chaines sont resolues **pour un projet donne** : si son volume d'entree est
la phase solide (le cas normal, convention `True = solide`), on prefixe un
changement de phase.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PRESETS", "build_presets", "FLUID"]

#: Marqueur remplace par le nom reel du calque de phase fluide.
FLUID = "{fluide}"

PRESETS: list[dict[str, Any]] = [
    {
        "id": "granulo",
        "label": "granulométrie",
        "summary": "Distance à la paroi, carte d'ouverture, distribution de taille de pore.",
        "requires": [],
        "steps": [
            {"step": "distance_transform", "params": {"out": "distance"}, "input": FLUID},
            {"step": "aperture_map", "params": {"out": "ouverture", "n_radii": 24}, "input": FLUID},
            {
                "step": "pore_size_distribution",
                "params": {"out": "granulometrie", "bins": 30, "mask": "@" + FLUID},
                "input": "ouverture",
            },
        ],
    },
    {
        "id": "cellules",
        "label": "cellules & cols",
        "summary": "Boules maximales, marqueurs, ligne de partage des eaux, morphométrie, cols.",
        "requires": [],
        "steps": [
            {"step": "distance_transform", "params": {"out": "distance"}, "input": FLUID},
            {
                "step": "cell_markers",
                "params": {"out": "marqueurs", "distance": "@distance", "fill_ratio": 0.55},
                "input": FLUID,
            },
            {
                "step": "watershed_cells",
                "params": {
                    "out": "cellules",
                    "markers": "@marqueurs",
                    "mask": "@" + FLUID,
                },
                # le relief est la DISTANCE, pas les marqueurs
                "input": "distance",
            },
            {"step": "cell_morphometry", "params": {"out": "morphometrie"}, "input": "cellules"},
            {"step": "throats", "params": {"out": "cols"}, "input": "cellules"},
        ],
    },
    {
        "id": "plateau",
        "label": "squelette de Plateau",
        "summary": "Nœuds et brins par la loi de Plateau, à partir des cellules segmentées.",
        "requires": ["cellules"],
        "steps": [
            {
                "step": "plateau_skeleton",
                "params": {"out": "plateau", "cells": "@cellules"},
                "input": "volume",
            },
        ],
    },
    {
        "id": "drainage",
        "label": "drainage",
        "summary": "Intrusion morphologique par la face z = 0 et courbe de rétention.",
        "requires": [],
        "steps": [
            {
                "step": "drainage",
                "params": {
                    "out": "drainage",
                    "method": "hilpert",
                    "step": 0.5,
                    "surface_tension": 0.0728,
                },
                "input": FLUID,
            },
        ],
    },
]


def build_presets(project) -> list[dict[str, Any]]:
    """Resout les chaines types pour un projet : phase fluide, calques presents.

    Rend une copie ou `FLUID` est remplace par le nom du calque de phase fluide,
    precedee au besoin d'une etape `complement`. Chaque chaine porte `available`
    et `missing` pour que l'interface sache ce qu'elle peut lancer.
    """
    layers = set(project.layers)
    phase = (project.phase or "solid").lower()

    if phase == "fluid":
        fluid_name, prefix = "volume", []
    else:
        fluid_name = "fluide"
        prefix = [{"step": "complement", "params": {"out": "fluide"}, "input": "volume"}]

    out: list[dict[str, Any]] = []
    for preset in PRESETS:
        steps = [_resolve(s, fluid_name) for s in preset["steps"]]
        needs_fluid = any(
            s.get("input") == fluid_name or "@" + fluid_name in str(s.get("params", {}))
            for s in steps
        )
        if needs_fluid and fluid_name not in layers:
            steps = [dict(s) for s in prefix] + steps
        missing = [r for r in preset["requires"] if r not in layers]
        out.append(
            {
                **{k: v for k, v in preset.items() if k != "steps"},
                "steps": steps,
                "missing": missing,
                "available": not missing,
            }
        )
    return out


def _resolve(step: dict[str, Any], fluid_name: str) -> dict[str, Any]:
    params = {
        k: (v.replace(FLUID, fluid_name) if isinstance(v, str) else v)
        for k, v in (step.get("params") or {}).items()
    }
    resolved = {"step": step["step"], "params": params}
    if step.get("input"):
        resolved["input"] = step["input"].replace(FLUID, fluid_name)
    return resolved

"""Enchainement declaratif d'etapes, sans interface graphique.

iMorph ne savait rien faire sans sa fenetre : chaque calcul etait un
`CalcModule` couple a un `QThread`, un `Param*Window` et la `CentralWidget`.
Ici, une analyse complete se decrit en donnees et s'execute d'un appel, depuis
un script, un notebook, un job de calcul ou la ligne de commande.

    from morphanalyzer.pipeline import Pipeline

    result = (
        Pipeline()
        .step("threshold_otsu")
        .step("keep_largest_component", connectivity=26)
        .step("porosity", out="porosity")
        .step("specific_surface", out="Sv")
        .run(volume)
    )
    print(result["porosity"], result["Sv"])

Equivalent en YAML, pour un traitement par lot reproductible :

    steps:
      - threshold_otsu
      - {keep_largest_component: {connectivity: 26}}
      - {porosity: {out: porosity}}
      - {specific_surface: {out: Sv}}
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Pipeline", "register", "available_steps", "run_from_config"]

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str, fn: Callable[..., Any] | None = None):
    """Enregistre une etape utilisable par nom. Utilisable en decorateur."""
    if fn is None:

        def deco(f: Callable[..., Any]):
            _REGISTRY[name] = f
            return f

        return deco
    _REGISTRY[name] = fn
    return fn


def available_steps() -> list[str]:
    """Noms des etapes disponibles, par ordre alphabetique."""
    return sorted(_REGISTRY)


@dataclass(slots=True)
class Step:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    out: str | None = None


@dataclass(slots=True)
class Pipeline:
    """Suite d'etapes appliquees a un volume.

    Chaque etape recoit le volume courant en premier argument. Si elle rend un
    volume (ou un tableau), il devient le volume courant ; sinon le resultat est
    range dans le contexte sous la cle `out` (ou le nom de l'etape).
    """

    steps: list[Step] = field(default_factory=list)
    verbose: bool = True

    def step(self, name: str, *, out: str | None = None, **params) -> Pipeline:
        if name not in _REGISTRY:
            raise KeyError(
                f"etape inconnue : {name!r}. Disponibles : {', '.join(available_steps()) or '(aucune)'}"
            )
        self.steps.append(Step(name, params, out))
        return self

    def run(self, volume, context: dict[str, Any] | None = None) -> dict[str, Any]:
        import numpy as np

        from morphanalyzer.core import Volume

        ctx: dict[str, Any] = dict(context or {})
        current = volume
        ctx["input"] = volume
        for s in self.steps:
            fn = _REGISTRY[s.name]
            t0 = time.perf_counter()
            res = fn(current, **s.params)
            dt = time.perf_counter() - t0
            if isinstance(res, (Volume, np.ndarray)):
                current = res
                ctx[s.out or s.name] = res
            else:
                ctx[s.out or s.name] = res
            if self.verbose:
                shown = res if not isinstance(res, (Volume, np.ndarray)) else type(res).__name__
                print(f"[{s.name}] {dt:.2f} s -> {shown}")
        ctx["output"] = current
        return ctx


def run_from_config(config: dict | str | Path, volume) -> dict[str, Any]:
    """Execute un pipeline decrit par un dict ou un fichier YAML/JSON."""
    if isinstance(config, (str, Path)):
        p = Path(config)
        text = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            from morphanalyzer._deps import require

            yaml = require("yaml", reason="la lecture de pipelines YAML")
            config = yaml.safe_load(text)
        else:
            import json

            config = json.loads(text)
    assert isinstance(config, dict)
    pipe = Pipeline(verbose=config.get("verbose", True))
    for entry in config.get("steps", []):
        if isinstance(entry, str):
            pipe.step(entry)
        elif isinstance(entry, dict) and len(entry) == 1:
            name, params = next(iter(entry.items()))
            params = dict(params or {})
            pipe.step(name, out=params.pop("out", None), **params)
        else:
            raise ValueError(f"etape mal formee dans la configuration : {entry!r}")
    return pipe.run(volume)


def _register_builtin() -> None:
    """Enregistre les etapes des modules deja implementes."""
    from morphanalyzer import (
        cortical,
        distance,
        filters,
        granulometry,
        mesh,
        metrics,
        network,
        segmentation,
        shape,
        skeleton,
        tortuosity,
    )

    register("threshold_otsu", filters.threshold_otsu)
    register("threshold_value", filters.threshold_value)
    register("threshold_hysteresis", filters.threshold_hysteresis)
    register("median", filters.median)
    register("denoise_nl_means", filters.denoise_nl_means)
    register("erode", filters.erode)
    register("dilate", filters.dilate)
    register("open_binary", filters.open_binary)
    register("close_binary", filters.close_binary)
    register("keep_largest_component", filters.keep_largest_component)
    register("remove_small_objects", filters.remove_small_objects)
    register("fill_holes", filters.fill_holes)
    register("hessian_features", filters.hessian_features)
    register("porosity", metrics.porosity)
    register("porosity_per_slice", metrics.porosity_per_slice)
    register("specific_surface", metrics.specific_surface)
    register("representative_volume", metrics.representative_volume)
    register("distance_transform", distance.distance_transform)
    register("aperture_map", granulometry.aperture_map)
    register("skeletonize", skeleton.skeletonize)
    register("distance_ridge", skeleton.distance_ridge)
    register("shape_classification", shape.shape_classification)
    register("maximal_balls", granulometry.maximal_balls)
    register("cell_markers", granulometry.cell_markers)
    register("cell_morphometry", segmentation.cell_morphometry)
    register("throats", segmentation.throats)
    register("connectivity", segmentation.connectivity)
    register("pore_network", segmentation.pore_network)
    register("plateau_skeleton", skeleton.plateau_skeleton)
    register("travel_time", distance.travel_time)
    register("geodesic_distance", distance.geodesic_distance)
    register("point_tortuosity", tortuosity.point_tortuosity)
    register("plane_tortuosity", tortuosity.plane_tortuosity)
    register("directional_tortuosity", tortuosity.directional_tortuosity)
    register("poiseuille_tortuosity", tortuosity.poiseuille_tortuosity)
    register("surface_mesh", mesh.surface_mesh)
    register("save_mesh", mesh.save_mesh)
    register("drainage", network.drainage)
    register("saturation_curve", network.saturation_curve)
    register("invasion_percolation", network.invasion_percolation)
    register("capillary_pressure", network.capillary_pressure)
    register("angular_profile", cortical.angular_profile)
    register("angular_aperture", cortical.angular_aperture)
    register("radial_profile", cortical.radial_profile)
    register("cortical_connectivity", cortical.cortical_connectivity)
    register("voronoi_2d", cortical.voronoi_2d)


_register_builtin()

"""Dossier de projet : un echantillon, ses calques, ses tables, son historique.

iMorph rangeait tout dans une base XML propriaire (`iMorphDataBase.xml`) et des
`.bin` maison, avec une hierarchie bibliotheque / echantillon / resolution /
ROI / phase. Impossible a lire sans le logiciel, impossible a versionner.

Ici un projet est **un dossier ordinaire** :

    mon_echantillon/
        morphanalyzer.json        manifeste : taille de voxel, calques, historique
        layers/volume.npy         le volume d'entree
        layers/distance.npy       les cartes calculees
        layers/cells.npy
        tables/pore_size.csv      les courbes et tableaux
        exports/                  ce qu'on sort pour un article

Chaque fichier est lisible sans morphanalyzer : `.npy` pour les tableaux, JSON
pour le manifeste, CSV pour les tables. Un projet ouvert par l'interface reste
utilisable depuis un script, et reciproquement.

Ce module n'importe **rien** de graphique : il est le modele de donnees partage
entre la ligne de commande, les notebooks et l'interface web.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from morphanalyzer.core import Volume
from morphanalyzer.core.volume import as_array

__all__ = ["Project", "Layer", "MANIFEST_NAME", "KINDS"]

MANIFEST_NAME = "morphanalyzer.json"
FORMAT_VERSION = 1

#: Nature d'un calque. Elle decide de la facon de l'afficher, pas de son dtype.
KINDS = ("grey", "binary", "labels", "scalar")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _infer_kind(data: np.ndarray) -> str:
    if data.dtype == bool:
        return "binary"
    if data.dtype.kind in "fc":
        return "scalar"
    return "labels"


@dataclass
class Layer:
    """Un tableau 3D range dans le projet.

    `kind` dit comment l'afficher, pas ce que contient le tableau :

    - `grey` : niveaux de gris d'origine, fenetrage libre ;
    - `binary` : masque, deux couleurs ;
    - `labels` : etiquettes entieres, une couleur par objet ;
    - `scalar` : champ continu (distance, ouverture, rayon d'envahissement),
      rampe a une seule teinte. La valeur `nodata` est rendue transparente.
    """

    name: str
    kind: str
    dtype: str
    shape: tuple[int, int, int]
    file: str
    vmin: float | None = None
    vmax: float | None = None
    nodata: float | None = None
    n_labels: int | None = None
    description: str = ""
    step: str | None = None
    created: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Layer:
        d = dict(d)
        d["shape"] = tuple(int(v) for v in d["shape"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class Project:
    """Un dossier de projet, ouvert en lecture-ecriture.

    Les calques sont **memmappes** : ouvrir un projet ne charge rien en memoire,
    et une coupe ne lit que la coupe.

        proj = Project.create("mousse/", volume=vol, name="Recemat 1723")
        proj.add_layer("distance", dist, description="distance a la paroi")
        proj.layer("distance")[64]          # une coupe, sans tout charger
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._manifest: dict[str, Any] = {}
        self._cache: dict[str, np.ndarray] = {}
        self._load_manifest()

    # -- ouverture et creation --------------------------------------------
    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        volume=None,
        name: str | None = None,
        voxel_size=None,
        unit: str = "um",
        source: str | None = None,
        exist_ok: bool = False,
    ) -> Project:
        """Cree un dossier de projet, avec eventuellement son volume d'entree."""
        path = Path(path)
        if (path / MANIFEST_NAME).exists() and not exist_ok:
            raise FileExistsError(
                f"{path} contient deja un projet morphanalyzer ; "
                "passer exist_ok=True pour le reouvrir"
            )
        (path / "layers").mkdir(parents=True, exist_ok=True)
        (path / "tables").mkdir(exist_ok=True)
        (path / "exports").mkdir(exist_ok=True)

        if voxel_size is None:
            voxel_size = volume.voxel_size if isinstance(volume, Volume) else (1.0, 1.0, 1.0)
        if np.isscalar(voxel_size):
            voxel_size = (float(voxel_size),) * 3
        if isinstance(volume, Volume):
            unit = volume.unit

        manifest = {
            "format": FORMAT_VERSION,
            "name": name or path.name,
            "created": _now(),
            "voxel_size": [float(v) for v in voxel_size],
            "unit": unit,
            "source": source,
            "layers": {},
            "tables": {},
            "history": [],
        }
        (path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        proj = cls(path)
        if volume is not None:
            arr = as_array(volume)
            proj.add_layer(
                "volume",
                arr,
                kind="binary" if arr.dtype == bool else "grey",
                description="volume d'entree",
            )
        return proj

    @classmethod
    def open(cls, path: str | Path) -> Project:
        path = Path(path)
        if not (path / MANIFEST_NAME).exists():
            raise FileNotFoundError(
                f"{path} n'est pas un projet morphanalyzer (pas de {MANIFEST_NAME})"
            )
        return cls(path)

    def _load_manifest(self) -> None:
        f = self.path / MANIFEST_NAME
        if f.exists():
            self._manifest = json.loads(f.read_text())

    def save(self) -> None:
        """Reecrit le manifeste. Les calques sont ecrits a l'ajout."""
        (self.path / MANIFEST_NAME).write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False)
        )

    # -- metadonnees -------------------------------------------------------
    @property
    def name(self) -> str:
        return self._manifest.get("name", self.path.name)

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        return tuple(self._manifest.get("voxel_size", (1.0, 1.0, 1.0)))  # type: ignore[return-value]

    @property
    def unit(self) -> str:
        return self._manifest.get("unit", "um")

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._manifest.get("history", []))

    @property
    def shape(self) -> tuple[int, int, int] | None:
        for layer in self.layers.values():
            return layer.shape
        return None

    # -- calques -----------------------------------------------------------
    @property
    def layers(self) -> dict[str, Layer]:
        return {k: Layer.from_dict(v) for k, v in self._manifest.get("layers", {}).items()}

    def add_layer(
        self,
        name: str,
        data,
        *,
        kind: str | None = None,
        description: str = "",
        step: str | None = None,
        nodata: float | None = None,
        overwrite: bool = True,
    ) -> Layer:
        """Range un tableau 3D dans le projet et l'inscrit au manifeste."""
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"nom de calque invalide : {name!r}")
        arr = as_array(data)
        if name in self._manifest.get("layers", {}) and not overwrite:
            raise FileExistsError(f"le calque {name!r} existe deja")
        kind = kind or _infer_kind(arr)
        if kind not in KINDS:
            raise ValueError(f"kind doit etre l'un de {KINDS}, recu {kind!r}")

        rel = f"layers/{name}.npy"
        (self.path / "layers").mkdir(parents=True, exist_ok=True)
        np.save(self.path / rel, arr)

        vmin = vmax = None
        n_labels = None
        if kind in ("grey", "scalar"):
            finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
            if nodata is not None:
                finite = finite[finite != nodata]
            if finite.size:
                vmin, vmax = float(finite.min()), float(finite.max())
        elif kind == "labels":
            n_labels = int(arr.max())

        layer = Layer(
            name=name,
            kind=kind,
            dtype=str(arr.dtype),
            shape=tuple(int(v) for v in arr.shape),
            file=rel,
            vmin=vmin,
            vmax=vmax,
            nodata=nodata,
            n_labels=n_labels,
            description=description,
            step=step,
        )
        self._manifest.setdefault("layers", {})[name] = layer.to_dict()
        self._cache.pop(name, None)
        self.save()
        return layer

    def layer(self, name: str) -> np.ndarray:
        """Le tableau du calque, **memmappe** : rien n'est lu avant l'acces."""
        if name in self._cache:
            return self._cache[name]
        info = self._manifest.get("layers", {}).get(name)
        if info is None:
            raise KeyError(f"calque inconnu : {name!r}. Disponibles : {list(self.layers)}")
        arr = np.load(self.path / info["file"], mmap_mode="r")
        self._cache[name] = arr
        return arr

    def volume(self, name: str = "volume") -> Volume:
        """Le calque, enveloppe dans un `Volume` avec sa taille de voxel."""
        info = self.layers[name]
        return Volume(
            np.asarray(self.layer(name)),
            voxel_size=self.voxel_size,
            unit=self.unit,
            name=f"{self.name}/{name}",
            meta={"kind": info.kind, "project": str(self.path)},
        )

    def remove_layer(self, name: str) -> None:
        info = self._manifest.get("layers", {}).pop(name, None)
        if info is None:
            raise KeyError(f"calque inconnu : {name!r}")
        self._cache.pop(name, None)
        f = self.path / info["file"]
        if f.exists():
            f.unlink()
        self.save()

    # -- tables ------------------------------------------------------------
    @property
    def tables(self) -> dict[str, dict[str, Any]]:
        return dict(self._manifest.get("tables", {}))

    def add_table(
        self, name: str, frame: pd.DataFrame, *, description: str = "", step: str | None = None
    ) -> dict[str, Any]:
        """Range un tableau de resultats en CSV, lisible par n'importe quoi."""
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"nom de table invalide : {name!r}")
        (self.path / "tables").mkdir(parents=True, exist_ok=True)
        rel = f"tables/{name}.csv"
        frame.to_csv(self.path / rel, index=False)
        info = {
            "name": name,
            "file": rel,
            "columns": [str(c) for c in frame.columns],
            "n_rows": int(len(frame)),
            "description": description,
            "step": step,
            "created": _now(),
        }
        self._manifest.setdefault("tables", {})[name] = info
        self.save()
        return info

    def table(self, name: str) -> pd.DataFrame:
        info = self._manifest.get("tables", {}).get(name)
        if info is None:
            raise KeyError(f"table inconnue : {name!r}. Disponibles : {list(self.tables)}")
        return pd.read_csv(self.path / info["file"])

    # -- historique --------------------------------------------------------
    def log_step(
        self,
        step: str,
        params: dict[str, Any],
        *,
        outputs: list[str] | None = None,
        duration: float | None = None,
        note: str = "",
    ) -> None:
        """Inscrit une etape executee, pour que le projet raconte ce qu'il a subi."""
        self._manifest.setdefault("history", []).append(
            {
                "step": step,
                "params": {k: _jsonable(v) for k, v in params.items()},
                "outputs": outputs or [],
                "duration": duration,
                "note": note,
                "at": _now(),
            }
        )
        self.save()

    def to_pipeline_config(self) -> dict[str, Any]:
        """Rejoue l'historique sous forme de configuration de pipeline.

        Ce que l'interface a lance a la souris redevient un fichier YAML
        executable en lot — c'est tout l'interet de garder l'historique.
        """
        steps = []
        for entry in self.history:
            params = dict(entry.get("params", {}))
            out = entry.get("outputs") or []
            if out:
                params["out"] = out[0]
            steps.append({entry["step"]: params} if params else entry["step"])
        return {"steps": steps}

    def __repr__(self) -> str:  # pragma: no cover - agrement
        n = len(self.layers)
        t = len(self.tables)
        return f"<Project {self.name!r} : {n} calque(s), {t} table(s), {self.path}>"


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 64 else f"<ndarray {v.shape}>"
    return repr(v)

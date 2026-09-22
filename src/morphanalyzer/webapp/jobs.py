"""Execution d'un pipeline en tache de fond, avec suivi et persistance.

Chaque etape ecrit son resultat dans le projet : un tableau devient un calque,
un `DataFrame` devient une table, un scalaire va dans le journal. L'historique
du projet enregistre ce qui a ete lance, ce qui permet de **rejouer a la ligne
de commande ce qu'on a fait a la souris** (`Project.to_pipeline_config`).
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from morphanalyzer.core.volume import Volume, as_array
from morphanalyzer.pipeline import get_step, resolve_params
from morphanalyzer.project import Project

__all__ = ["Job", "JobRunner", "unpack"]


@dataclass
class Job:
    id: str
    steps: list[dict[str, Any]]
    input_layer: str
    status: str = "pending"  # pending | running | done | error
    current: int = -1
    log: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    started: float | None = None
    finished: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "current": self.current,
            "n_steps": len(self.steps),
            "steps": [s.get("step") for s in self.steps],
            "input_layer": self.input_layer,
            "log": self.log,
            "results": self.results,
            "outputs": self.outputs,
            "tables": self.tables,
            "warnings": self.warnings,
            "error": self.error,
            "elapsed": (self.finished or time.time()) - self.started if self.started else 0.0,
        }


class JobRunner:
    """Une file a un seul fil : les calculs lourds ne se marchent pas dessus."""

    def __init__(self, project: Project, lock: threading.Lock | None = None) -> None:
        self.project = project
        self.lock = lock or threading.Lock()
        self.jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._worker: threading.Thread | None = None
        self._queue: list[str] = []
        self._cv = threading.Condition()

    # -- API ---------------------------------------------------------------
    def submit(self, steps: list[dict[str, Any]], input_layer: str = "volume") -> Job:
        job = Job(id=uuid.uuid4().hex[:12], steps=list(steps), input_layer=input_layer)
        self.jobs[job.id] = job
        self._order.append(job.id)
        with self._cv:
            self._queue.append(job.id)
            self._cv.notify()
        self._ensure_worker()
        return job

    def run_sync(self, steps: list[dict[str, Any]], input_layer: str = "volume") -> Job:
        """Meme chose, mais sans fil : pratique pour les tests et les scripts."""
        job = Job(id=uuid.uuid4().hex[:12], steps=list(steps), input_layer=input_layer)
        self.jobs[job.id] = job
        self._order.append(job.id)
        self._execute(job)
        return job

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        return [self.jobs[i].to_dict() for i in self._order[-n:][::-1]]

    # -- mecanique ---------------------------------------------------------
    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._loop, name="morphanalyzer-jobs", daemon=True)
        self._worker.start()

    def _loop(self) -> None:  # pragma: no cover - exerce par le serveur
        while True:
            with self._cv:
                while not self._queue:
                    if not self._cv.wait(timeout=30.0):
                        return
                job_id = self._queue.pop(0)
            self._execute(self.jobs[job_id])

    def _execute(self, job: Job) -> None:
        job.status = "running"
        job.started = time.time()
        try:
            current: Any = np.asarray(self.project.layer(job.input_layer))
            for i, entry in enumerate(job.steps):
                job.current = i
                name = entry["step"]
                raw = dict(entry.get("params") or {})
                out_name = raw.pop("out", None) or name
                source = entry.get("input") or raw.pop("input", None)
                if source:
                    current = np.asarray(self.project.layer(source))
                fn = get_step(name)
                params = resolve_params(raw, self.project)
                t0 = time.perf_counter()
                # Les avertissements sont captures et remontes a l'interface. Sans cela
                # un repli silencieux — typiquement « numba absent », qui redonne le
                # watershed a relief quantifie — passerait inapercu dans un terminal
                # qu'on ne regarde pas.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    res = fn(current, **params)
                dt = time.perf_counter() - t0
                notes = [str(w.message) for w in caught]
                current, kept = self._store(
                    job, name, out_name, res, raw, dt, current, notes, source
                )
                entry = {
                    "step": name,
                    "params": raw,
                    "input": source,
                    "seconds": round(dt, 3),
                    "produced": kept,
                }
                if notes:
                    entry["warnings"] = notes
                    job.warnings.extend(notes)
                job.log.append(entry)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - on veut tout rapporter a l'interface
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.append({"step": "erreur", "traceback": traceback.format_exc(limit=6)})
        finally:
            job.current = len(job.steps)
            job.finished = time.time()

    def _store(
        self,
        job: Job,
        step: str,
        out_name: str,
        res: Any,
        params: dict,
        dt: float,
        current,
        notes: list[str] | None = None,
        source: str | None = None,
    ):
        """Range le resultat d'une etape selon sa nature.

        Une etape qui ne rend pas de volume 3D — une porosite, une table — ne
        change pas le volume courant : la suivante repart du meme tableau. Meme
        regle que `Pipeline.run`.

        Les objets composites des modules avances (un `DrainageResult`, un
        `PlateauSkeleton`) sont **decomposes** : leurs cartes deviennent des
        calques, leurs tables des tables. Sans cela l'interface n'aurait rien a
        montrer d'un drainage.
        """
        produced: list[str] = []
        names: list[str] = []  # tout ce qui a ete ecrit, calque ou table
        new_current = current
        with self.lock:
            for kind, name, value in unpack(res, out_name):
                if kind == "layer":
                    arr = as_array(value)
                    self.project.add_layer(name, arr, step=step)
                    job.outputs.append(name)
                    names.append(name)
                    produced.append(f"calque « {name} »")
                    if new_current is current:
                        new_current = arr
                elif kind == "table":
                    frame = value if isinstance(value, pd.DataFrame) else value.to_frame()
                    self.project.add_table(name, frame, step=step)
                    job.tables.append(name)
                    names.append(name)
                    produced.append(f"table « {name} » ({len(frame)} lignes)")
                else:
                    job.results[name] = value
                    names.append(name)
                    produced.append(f"{name} = {value}")
            note = "; ".join(produced) or "aucune sortie"
            if notes:
                note += " | avertissement : " + " | ".join(notes)
            self.project.log_step(
                step,
                params,
                outputs=names,
                duration=dt,
                note=note,
                input=source,
            )
        return new_current, ", ".join(produced) or "—"

    def __getitem__(self, job_id: str) -> Job:
        return self.jobs[job_id]


def unpack(res: Any, out: str) -> list[tuple[str, str, Any]]:
    """Decompose le resultat d'une etape en calques, tables et valeurs.

    Rend une liste de `(nature, nom, valeur)` avec `nature` dans
    `{"layer", "table", "value"}`. Les noms derivent de `out`, de sorte qu'une
    seule etape puisse produire plusieurs sorties nommees de facon previsible.
    """
    if isinstance(res, (Volume, np.ndarray)) and np.asarray(res).ndim == 3:
        return [("layer", out, res)]
    if isinstance(res, pd.DataFrame):
        return [("table", out, res)]
    if isinstance(res, pd.Series):
        return [("table", out, res.rename(out).reset_index())]

    cls = type(res).__name__
    if cls == "DrainageResult":
        return [("layer", out, res.filling_radius), ("table", f"{out}_courbe", res.curve)]
    if cls == "InvasionResult":
        return [
            ("table", f"{out}_cellules", res.cells.reset_index()),
            ("table", f"{out}_courbe", res.curve),
            ("value", f"{out}_saturation", round(float(res.final_saturation), 4)),
        ]
    if cls == "PlateauSkeleton":
        out_list = [
            ("layer", f"{out}_noeuds", res.node_mask),
            ("layer", f"{out}_brins", res.strut_mask),
            ("table", f"{out}_table_noeuds", _stringify(res.nodes)),
        ]
        if getattr(res, "edges", None) is not None and len(res.edges):
            out_list.append(("table", f"{out}_table_brins", res.edges))
        return out_list
    if cls == "ShapeTensor":
        return [("table", out, res.to_frame())]
    if cls == "TortuosityResult":
        return [
            ("value", out, round(float(res.value), 5)),
            ("value", f"{out}_ecart_type", round(float(res.std), 5)),
            ("layer", f"{out}_temps", res.travel_time),
        ]
    if cls == "ConnectivityProfile":
        return [("table", out, res.table)]
    if cls == "SectorProfile":
        return [("table", out, res.table.drop(columns=["_num"], errors="ignore"))]
    if cls == "Mesh":
        return [
            ("value", f"{out}_triangles", int(len(res.faces))),
            ("value", f"{out}_aire", round(float(res.area), 3)),
            ("value", f"{out}_volume", round(float(res.volume), 3)),
        ]
    if isinstance(res, tuple) and res and all(isinstance(x, pd.DataFrame) for x in res):
        names = ["cellules", "cols", "c", "d"]
        return [("table", f"{out}_{names[i]}", x) for i, x in enumerate(res)]
    return [("value", out, _scalar(res))]


def _stringify(frame: pd.DataFrame) -> pd.DataFrame:
    """Rend un DataFrame ecrivable en CSV : les colonnes de tuples deviennent du texte."""
    out = frame.copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, (tuple, list, set))).any():
            out[col] = out[col].map(
                lambda v: " ".join(map(str, v)) if isinstance(v, (tuple, list, set)) else v
            )
    return out


def _scalar(res: Any) -> Any:
    if isinstance(res, (bool, int, float, str)) or res is None:
        return res
    if isinstance(res, np.generic):
        return res.item()
    if isinstance(res, np.ndarray):
        return {"shape": list(res.shape), "dtype": str(res.dtype)}
    if hasattr(res, "to_dict"):
        try:
            return {k: _scalar(v) for k, v in res.to_dict().items()}  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    return repr(res)

"""Serveur de l'interface : une API JSON, et des coupes en PNG.

L'interface d'iMorph etait indissociable du calcul : chaque module etait un
`CalcModule` couple a un `QThread`, une fenetre de parametres et le widget
central. Ici c'est l'inverse — le serveur ne fait qu'exposer ce que la
bibliotheque sait deja faire :

    GET  /api/project              le manifeste, les calques, les tables
    GET  /api/slice.png?spec=...   une coupe composee, cote serveur
    GET  /api/value?z=&y=&x=       la valeur de chaque calque sous le curseur
    GET  /api/histogram?layer=     de quoi regler un fenetrage
    GET  /api/table/{nom}          une courbe ou un tableau de resultats
    GET  /api/steps                les etapes de pipeline disponibles
    POST /api/run                  lance un pipeline en tache de fond
    GET  /api/jobs[/{id}]          son avancement

Rien ici n'est indispensable : tout passe par `Project`, `pipeline` et
`render`, qui s'utilisent aussi bien depuis un script.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from morphanalyzer.project import Project
from morphanalyzer.webapp import render as R
from morphanalyzer.webapp.jobs import JobRunner

__all__ = ["create_app", "serve"]

STATIC = Path(__file__).parent / "static"


def _layer_views(project: Project, spec: dict[str, Any]) -> list[R.LayerView]:
    views: list[R.LayerView] = []
    known = project.layers
    for entry in spec.get("layers", []):
        name = entry.get("name")
        if name not in known:
            continue
        info = known[name]
        views.append(
            R.LayerView(
                data=project.layer(name),
                kind=entry.get("kind") or info.kind,
                ramp=entry.get("ramp") or R.DEFAULT_RAMP,
                vmin=_num(entry.get("vmin"), info.vmin),
                vmax=_num(entry.get("vmax"), info.vmax),
                nodata=_num(entry.get("nodata"), info.nodata),
                bands=_int(entry.get("bands")),
                nodata_color=entry.get("nodata_color") or None,
                alpha=float(entry.get("alpha", 1.0)),
                color=entry.get("color") or "#2a78d6",
                visible=bool(entry.get("visible", True)),
            )
        )
    return views


def _int(value):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _num(value, fallback):
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def create_app(project_path: str | Path, *, read_only: bool = False, create: bool = False):
    """Construit l'application FastAPI autour d'un dossier de projet.

    Le dossier doit deja etre un projet : servir un dossier quelconque en y
    creant un projet vide au passage serait une surprise desagreable. Passer
    `create=True` pour l'autoriser explicitement.
    """
    from morphanalyzer._deps import require

    fastapi = require("fastapi", reason="l'interface web")
    from fastapi import Body, HTTPException, Query
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    # `from __future__ import annotations` transforme les annotations en chaines,
    # que FastAPI resout dans les *globales du module*. Les classes importees ici
    # sont locales a la fonction : on annote donc uniquement avec des types
    # standards, et on passe le corps de requete par `Body`.

    path = Path(project_path)
    if (path / "morphanalyzer.json").exists():
        project = Project.open(path)
    elif create:
        project = Project.create(path)
    else:
        raise FileNotFoundError(
            f"{path} n'est pas un projet morphanalyzer (pas de morphanalyzer.json). "
            f"Le creer d'abord : morphanalyzer new {path} --volume <pile.tif> --voxel-size <n>"
        )
    lock = threading.Lock()
    runner = JobRunner(project, lock)

    app = fastapi.FastAPI(title="morphanalyzer", version="0.1", docs_url="/api/docs")
    app.state.project = project
    app.state.runner = runner
    app.state.read_only = read_only

    # -- description du projet --------------------------------------------
    @app.get("/api/project")
    def api_project() -> dict[str, Any]:
        project._load_manifest()
        return {
            "name": project.name,
            "path": str(project.path),
            "voxel_size": list(project.voxel_size),
            "unit": project.unit,
            "phase": project.phase,
            "shape": list(project.shape) if project.shape else None,
            "read_only": read_only,
            "layers": [layer.to_dict() for layer in project.layers.values()],
            "tables": list(project.tables.values()),
            "values": list(project.values.values()),
            "history": project.history,
            "ramps": sorted(R.RAMPS),
            "nodata_color": R.NODATA_COLOR,
        }

    # -- une coupe ---------------------------------------------------------
    @app.get("/api/slice.png")
    def api_slice(spec: str = Query(...)):
        try:
            parsed = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"spec illisible : {exc}") from exc
        views = _layer_views(project, parsed)
        if not views:
            raise HTTPException(404, "aucun calque connu dans la specification")
        try:
            rgba = R.render_slice(
                views,
                axis=int(parsed.get("axis", 0)),
                index=int(parsed.get("index", 0)),
                max_size=int(parsed.get("max_size", 2048)),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return Response(
            content=R.png_bytes(rgba),
            media_type="image/png",
            headers={"Cache-Control": "no-cache"},
        )

    # -- inspection --------------------------------------------------------
    @app.get("/api/value")
    def api_value(z: int, y: int, x: int) -> dict[str, Any]:
        out: dict[str, Any] = {"z": z, "y": y, "x": x, "values": {}}
        for name, info in project.layers.items():
            arr = project.layer(name)
            if not (0 <= z < arr.shape[0] and 0 <= y < arr.shape[1] and 0 <= x < arr.shape[2]):
                continue
            v = arr[z, y, x]
            out["values"][name] = {
                "value": None if isinstance(v, float) and not np.isfinite(v) else _py(v),
                "kind": info.kind,
            }
        return out

    @app.get("/api/histogram")
    def api_histogram(layer: str, bins: int = 64, sample: int = 4_000_000) -> dict[str, Any]:
        if layer not in project.layers:
            raise HTTPException(404, f"calque inconnu : {layer}")
        arr = np.asarray(project.layer(layer))
        flat = arr.reshape(-1)
        if flat.size > sample:  # sous-echantillonnage regulier
            flat = flat[:: max(1, flat.size // sample)]
        flat = flat[np.isfinite(flat)] if flat.dtype.kind == "f" else flat
        if flat.size == 0:
            return {"layer": layer, "edges": [], "counts": []}
        counts, edges = np.histogram(flat.astype(np.float64), bins=int(bins))
        return {
            "layer": layer,
            "edges": [float(v) for v in edges],
            "counts": [int(v) for v in counts],
            "vmin": float(flat.min()),
            "vmax": float(flat.max()),
        }

    # -- tables ------------------------------------------------------------
    @app.get("/api/table/{name}.csv")
    def api_table_csv(name: str):
        # declaree avant la route generique : sinon `{name}` avale « courbe.csv »
        info = project.tables.get(name)
        if info is None:
            raise HTTPException(404, f"table inconnue : {name}")
        return FileResponse(project.path / info["file"], filename=f"{name}.csv")

    @app.get("/api/table/{name}")
    def api_table(name: str) -> dict[str, Any]:
        try:
            frame = project.table(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        frame = frame.replace({np.nan: None})
        return {
            "name": name,
            "columns": [str(c) for c in frame.columns],
            "rows": frame.to_dict(orient="records"),
        }

    # -- pipeline ----------------------------------------------------------
    @app.get("/api/steps")
    def api_steps() -> list[dict[str, Any]]:
        from morphanalyzer.pipeline import available_steps, get_step, step_parameters

        out = []
        for name in available_steps():
            fn = get_step(name)
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            out.append(
                {
                    "name": name,
                    "summary": doc,
                    "module": fn.__module__.replace("morphanalyzer.", ""),
                    "parameters": step_parameters(name),
                }
            )
        return out

    @app.get("/api/presets")
    def api_presets(target: str = "fluid") -> list[dict[str, Any]]:
        from morphanalyzer.webapp.presets import build_presets

        project._load_manifest()
        try:
            return build_presets(project, target)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/run")
    def api_run(body: dict = Body(...)) -> dict[str, Any]:
        if read_only:
            raise HTTPException(403, "serveur en lecture seule")
        steps = body.get("steps") or []
        if not steps:
            raise HTTPException(400, "aucune etape a executer")
        job = runner.submit(steps, body.get("input_layer") or "volume")
        return job.to_dict()

    @app.get("/api/jobs")
    def api_jobs() -> list[dict[str, Any]]:
        return runner.recent()

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str) -> dict[str, Any]:
        try:
            return runner[job_id].to_dict()
        except KeyError as exc:
            raise HTTPException(404, f"tache inconnue : {job_id}") from exc

    @app.get("/api/pipeline.yaml")
    def api_pipeline():
        """L'historique du projet, en configuration rejouable en lot."""
        config = project.to_pipeline_config()
        try:
            import yaml

            text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        except ImportError:
            text = json.dumps(config, indent=2, ensure_ascii=False)
        return Response(text, media_type="text/plain; charset=utf-8")

    @app.delete("/api/layer/{name}")
    def api_delete_layer(name: str) -> dict[str, Any]:
        if read_only:
            raise HTTPException(403, "serveur en lecture seule")
        try:
            with lock:
                project.remove_layer(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"removed": name}

    # -- interface ---------------------------------------------------------
    if STATIC.is_dir():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    else:  # pragma: no cover - installation incomplete

        @app.get("/")
        def _missing():
            return JSONResponse({"error": "fichiers statiques absents"}, status_code=500)

    return app


def _py(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return round(v, 6)
    return v


def serve(
    project_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    read_only: bool = False,
    open_browser: bool = True,
) -> None:  # pragma: no cover - boucle serveur
    """Lance le serveur. `Ctrl-C` pour arreter."""
    from morphanalyzer._deps import require

    uvicorn = require("uvicorn", reason="l'interface web")
    app = create_app(project_path, read_only=read_only)
    url = f"http://{host}:{port}/"
    print(f"morphanalyzer — projet « {app.state.project.name} »")
    print(f"  interface : {url}")
    print(f"  API       : {url}api/docs")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")

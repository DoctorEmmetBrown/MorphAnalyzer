"""Ligne de commande.

Un complement au mode bibliotheque, pas un substitut : tout ce que fait la CLI
est accessible directement en Python.

    morphanalyzer info volume.tif --voxel-size 7.46
    morphanalyzer porosity volume.tif --otsu
    morphanalyzer phantom voronoi_foam --shape 128 --out foam.tif
    morphanalyzer run analyse.yaml volume.tif
    morphanalyzer steps
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["main"]


def _app():
    from morphanalyzer._deps import require

    typer = require("typer", reason="la ligne de commande")
    return typer, typer.Typer(
        add_completion=False, help="Analyse morphologique 3D (portage iMorph)"
    )


def main() -> None:  # pragma: no cover - point d'entree
    typer, app = _app()

    def _load(path: Path, voxel_size: float):
        from morphanalyzer import io

        p = Path(path)
        if p.is_dir() or p.suffix.lower() in (".tif", ".tiff"):
            return io.read_stack(p, voxel_size=voxel_size)
        raise typer.BadParameter(
            f"{p.name} : utiliser une pile TIFF ou un repertoire d'images ; "
            "pour du RAW, passer par l'API (morphanalyzer.io.read_raw) qui exige shape et dtype"
        )

    @app.command()
    def info(path: Path, voxel_size: float = 1.0) -> None:
        """Dimensions, dtype, histogramme resume."""

        vol = _load(path, voxel_size)
        a = vol.data
        typer.echo(f"{vol!r}")
        typer.echo(f"  min={a.min()}  max={a.max()}  moyenne={float(a.mean()):.4g}")
        typer.echo(
            f"  taille physique : {tuple(round(v, 2) for v in vol.physical_shape)} {vol.unit}"
        )

    @app.command()
    def porosity(
        path: Path,
        voxel_size: float = 1.0,
        otsu: bool = True,
        threshold: float | None = None,
        per_slice: bool = False,
    ) -> None:
        """Porosite totale (et par coupe)."""
        from morphanalyzer import filters, metrics

        vol = _load(path, voxel_size)
        if vol.dtype != bool:
            binv = (
                filters.threshold_otsu(vol)
                if threshold is None and otsu
                else filters.threshold_value(vol, threshold or 0)
            )
            typer.echo(f"seuil = {binv.meta.get('threshold')}")
        else:
            binv = vol
        typer.echo(f"porosite = {metrics.porosity(binv):.4f}")
        if per_slice:
            for k, p in enumerate(metrics.porosity_per_slice(binv)):
                typer.echo(f"  coupe {k:5d} : {p:.4f}")

    @app.command()
    def surface(path: Path, voxel_size: float = 1.0) -> None:
        """Surface specifique par marching cubes."""
        from morphanalyzer import filters, metrics

        vol = _load(path, voxel_size)
        binv = vol if vol.dtype == bool else filters.threshold_otsu(vol)
        typer.echo(f"surface specifique = {metrics.specific_surface(binv):.6g} {vol.unit}^-1")

    @app.command()
    def phantom(
        kind: str, shape: int = 128, out: Path = Path("phantom.tif"), seed: int = 0
    ) -> None:
        """Genere un volume synthetique a verite terrain connue."""
        from morphanalyzer import io, phantoms

        fn = getattr(phantoms, kind, None)
        if fn is None:
            raise typer.BadParameter(
                f"fantome inconnu : {kind}. Disponibles : {', '.join(phantoms.__all__)}"
            )
        kwargs = {"shape": (shape,) * 3}
        if "seed" in fn.__code__.co_varnames:
            kwargs["seed"] = seed
        vol = fn(**kwargs)
        io.write_stack(vol, out)
        typer.echo(f"ecrit : {out}")
        for k, v in vol.meta.get("truth", {}).items():
            if isinstance(v, (int, float, str, bool)):
                typer.echo(f"  {k} = {v}")

    @app.command()
    def steps() -> None:
        """Liste les etapes de pipeline disponibles."""
        from morphanalyzer.pipeline import available_steps

        for s in available_steps():
            typer.echo(s)

    @app.command()
    def run(config: Path, path: Path, voxel_size: float = 1.0) -> None:
        """Execute un pipeline YAML/JSON sur un volume."""
        from morphanalyzer.pipeline import run_from_config

        ctx = run_from_config(config, _load(path, voxel_size))
        for k, v in ctx.items():
            if isinstance(v, (int, float, str, bool)):
                typer.echo(f"{k} = {v}")

    app()


if __name__ == "__main__":  # pragma: no cover
    main()

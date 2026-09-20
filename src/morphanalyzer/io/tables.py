"""Tables de resultats : cellules, cols, brins, noeuds.

iMorph ecrivait des `.txt` maison et une base XML. Ici, un `DataFrame` par
famille d'objets, ecrit en Parquet (ou CSV a la demande) : directement
exploitable en analyse et versionnable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["read_table", "write_table"]


def write_table(df: pd.DataFrame, path: str | Path, *, meta: dict | None = None) -> Path:
    """Ecrit un `DataFrame`. `.parquet` par defaut, `.csv` si l'extension le dit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if meta:
        df = df.attrs and df or df
        df.attrs.update(meta)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        try:
            df.to_parquet(path, index=False)
        except ImportError:  # pragma: no cover
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)

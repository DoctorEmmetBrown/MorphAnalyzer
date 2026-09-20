#!/usr/bin/env python3
"""Inventaire reproductible d'une arborescence iMorph.

Repond a la seule question qui compte avant de porter quoi que ce soit :
**quel fichier est reellement compile ?** L'archive iMorph 3.2 contient 46
doublons (`*.bak`, `* - Copie.cpp`, `-old`, `-simon`) et plusieurs variantes de
travail d'un meme module. qmake tranche selon l'ordre des `VPATH` : pour chaque
nom declare, il retient la premiere occurrence trouvee. Ce script rejoue cette
resolution, et tout ce qui n'est pas retenu est du code mort.

    python tools/inventory.py /chemin/vers/iMorph3.2 --markdown docs/INVENTAIRE.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

CATEGORIES = [
    ("Gui", "interface"),
    ("DataType", "types de donnees"),
    ("FilterModules", "filtres"),
    ("PhysicalModules/iMorph_Rad", "radiatif"),
    ("PhysicalModules/PoreNetworkModelling", "reseau de pores"),
    ("Thread/Cortical", "cortical"),
    ("Thread/CrossSection", "sections"),
    ("Thread/Granulometry", "granulometrie / distance / morphometrie"),
    ("Thread/Mesh", "maillage"),
    ("Thread/Segment", "segmentation avancee"),
    ("Thread/ShapeClassif", "classification de forme"),
    ("Thread/Skeleton", "squelette"),
    ("Thread/Simplificator", "decimation"),
    ("Thread/Tortuosity", "tortuosite"),
    ("Thread/Import", "import"),
    ("Thread/Export", "export"),
]

# Artefacts generes par qmake/moc/rcc : ni source, ni code mort.
GENERATED = re.compile(r"^(moc_|qrc_|ui_)")

THIRD_PARTY = {
    "qcustomplot",
    "qrc_iMorph",
    "pictureflow",
    "qxtspanslider",
    "qxtpimpl",
    "qxtspanslider_p",
    "graphicViewZoom",
    "nlmeans_lib",
    "maxflow",
    "graphKolmo",
    "arcs",
    "block",
    "progmesh",
    "mersenneTwister",
    "triangulate",
    "boxTriangleOverlap",
    "heapReductor",
    "listReductor",
}

DUP = re.compile(r"(\s*[-(]\s*(Copie|Copy|copie|copy)\s*\)?|\.bak|\.orig|-old|_old|-simon)")


def parse_pro(pro_path: Path) -> tuple[list[str], list[str]]:
    """Rend (vpath ordonne, fichiers declares), en ignorant les lignes commentees."""
    vpath: list[str] = []
    declared: list[str] = []
    for line in pro_path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        m = re.match(r"VPATH\s*\+?=\s*(\S+)", s)
        if m:
            d = m.group(1).lstrip("./")
            if d and d not in vpath:
                vpath.append(d)
            continue
        declared.extend(re.findall(r"([\w\-]+\.(?:cpp|h|c))\b", s))
    return vpath, sorted(set(declared))


def analyse(root: Path) -> dict:
    pro = next(root.glob("*.pro"))
    vpath, declared = parse_pro(pro)

    def is_source(p: Path) -> bool:
        if not p.is_file() or GENERATED.match(p.name):
            return False
        if "Sources" not in p.parts:  # artefacts de build a la racine
            return False
        return p.suffix in (".cpp", ".h", ".c") or p.name.endswith((".bak", ".orig"))

    everywhere: dict[str, list[str]] = defaultdict(list)
    for p in root.rglob("*"):
        if is_source(p):
            everywhere[p.name].append(str(p.relative_to(root)))

    def resolve(name: str) -> str | None:
        for d in vpath:
            cand = root / d / name
            if cand.is_file():
                return str(cand.relative_to(root))
        return None

    built, missing, shadowed = {}, [], []
    for n in declared:
        r = resolve(n)
        if r is None:
            missing.append(n)
            continue
        built[n] = r
        for other in everywhere.get(n, []):
            if other != r:
                shadowed.append((n, r, other))

    built_paths = set(built.values())

    # Un en-tete absent du .pro reste vivant s'il est inclus quelque part :
    # qmake n'a pas besoin de declarer les headers pour que le code compile.
    # Sans cette verification, on classerait a tort des milliers de lignes
    # utiles en code mort.
    included: set[str] = set()
    for rel in built_paths:
        try:
            text = (root / rel).read_text(errors="ignore")
        except OSError:
            continue
        included.update(re.findall(r'#\s*include\s*"([^"]+)"', text))
    frontier = set(built_paths)
    for _ in range(12):  # cloture transitive des inclusions
        new_paths = set()
        for name in list(included):
            base = os.path.basename(name)
            r = resolve(base)
            if r and r not in built_paths and r not in new_paths:
                new_paths.add(r)
        if not new_paths:
            break
        for rel in new_paths:
            try:
                included.update(
                    re.findall(r'#\s*include\s*"([^"]+)"', (root / rel).read_text(errors="ignore"))
                )
            except OSError:
                pass
        built_paths |= new_paths
    del frontier

    dead, dup = [], []
    for p in sorted({str(x.relative_to(root)) for x in root.rglob("*") if is_source(x)}):
        if p in built_paths:
            continue
        (dup if DUP.search(os.path.basename(p)) else dead).append(p)

    def loc(rel: str) -> int:
        try:
            return sum(1 for _ in (root / rel).open(errors="ignore"))
        except OSError:
            return 0

    cats: dict[str, dict] = {}
    for prefix, label in CATEGORIES:
        files = [p for p in built_paths if p.startswith(f"Sources/{prefix}")]
        own = [p for p in files if Path(p).stem not in THIRD_PARTY]
        third = [p for p in files if Path(p).stem in THIRD_PARTY]
        cats[prefix] = {
            "label": label,
            "n_files": len(files),
            "loc": sum(loc(p) for p in own),
            "loc_third_party": sum(loc(p) for p in third),
        }

    return {
        "root": str(root),
        "pro": pro.name,
        "vpath": vpath,
        "n_declared": len(declared),
        "n_built": len(built),
        "missing": missing,
        "built": built,
        "shadowed": shadowed,
        "duplicates": dup,
        "dead_code": dead,
        "loc_duplicates": sum(loc(p) for p in dup),
        "loc_dead": sum(loc(p) for p in dead),
        "dead_loc": {p: loc(p) for p in dead},
        "categories": cats,
    }


def to_markdown(inv: dict) -> str:
    L = [
        "# Inventaire iMorph 3.2 — ce qui est reellement compile",
        "",
        f"Genere par `tools/inventory.py` depuis `{inv['pro']}`.",
        "",
        f"- fichiers declares dans le `.pro` : **{inv['n_declared']}**",
        f"- resolus dans l'arborescence : **{inv['n_built']}**",
        f"- introuvables : **{len(inv['missing'])}**"
        + (f" ({', '.join(inv['missing'])})" if inv["missing"] else ""),
        f"- doublons non compiles : **{len(inv['duplicates'])}** fichiers, "
        f"{inv['loc_duplicates']:,} lignes".replace(",", " "),
        f"- autres fichiers ni compiles ni inclus : **{len(inv['dead_code'])}** fichiers, "
        f"{inv['loc_dead']:,} lignes".replace(",", " "),
        "",
        "## Lignes par domaine (code iMorph seul, tiers exclus)",
        "",
        "| domaine | dossier | fichiers | lignes | dont tiers |",
        "|---|---|---:|---:|---:|",
    ]
    for prefix, c in inv["categories"].items():
        if not c["n_files"]:
            continue
        L.append(
            f"| {c['label']} | `{prefix}` | {c['n_files']} | {c['loc']:,} | {c['loc_third_party']:,} |".replace(
                ",", " "
            )
        )
    total = sum(c["loc"] for c in inv["categories"].values())
    tp = sum(c["loc_third_party"] for c in inv["categories"].values())
    L += [
        "",
        f"**Total compile : {total + tp:,} lignes**, dont {tp:,} de bibliotheques tierces.".replace(
            ",", " "
        ),
        "",
    ]

    if inv["shadowed"]:
        L += [
            "## Fichiers masques (meme nom dans deux dossiers du VPATH)",
            "",
            "qmake retient la premiere occurrence dans l'ordre des `VPATH`. "
            "La colonne « ignore » est du code mort, a ne pas porter.",
            "",
            "| nom | compile | ignore |",
            "|---|---|---|",
        ]
        seen = set()
        for n, r, o in inv["shadowed"]:
            if (n, o) in seen:
                continue
            seen.add((n, o))
            L.append(f"| `{n}` | `{r}` | `{o}` |")
        L.append("")

    if inv["dead_code"]:
        L += [
            "## Fichiers ni compiles ni inclus — a ne pas porter",
            "",
            "Ces fichiers ne figurent pas dans le `.pro` et aucun fichier compile ne les",
            "inclut : ce sont des experiences abandonnees ou des variantes de travail.",
            "Les plus volumineux d'abord.",
            "",
            "| lignes | fichier |",
            "|---:|---|",
        ]
        rows = sorted(((inv["dead_loc"][p], p) for p in inv["dead_code"]), reverse=True)
        L += [f"| {n} | `{p}` |" for n, p in rows if n >= 50]
        small = [p for n, p in rows if n < 50]
        if small:
            L += ["", f"Et {len(small)} fichiers de moins de 50 lignes."]
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="racine d'une arborescence iMorph (contenant le .pro)")
    ap.add_argument("--json", type=Path, help="ecrire l'inventaire brut en JSON")
    ap.add_argument("--markdown", type=Path, help="ecrire le rapport en Markdown")
    args = ap.parse_args()

    inv = analyse(args.root)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(inv, indent=1))
    md = to_markdown(inv)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)
    else:
        print(md)


if __name__ == "__main__":
    main()

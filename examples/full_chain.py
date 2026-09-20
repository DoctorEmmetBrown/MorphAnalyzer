#!/usr/bin/env python3
"""La chaine complete, de la binarisation au squelette — demonstration.

Enchaine les chapitres 2 et 3 de la these sur un fantome de Voronoi dont la
verite terrain est connue exactement, et affiche a chaque etape ce qui est mesure
et ce qui etait attendu.

    python examples/full_chain.py

Aucune interface graphique. Numba accelere le watershed s'il est installe.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage as ndi

import morphanalyzer as ma


def section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    t_total = time.perf_counter()

    section("0. Fantome — verite terrain connue")
    vol = ma.phantoms.voronoi_foam(
        shape=(128,) * 3, n_cells=40, strut=3.0, min_seed_gap=20.0, seed=7
    )
    truth = vol.meta["truth"]
    solid, fluid = vol.solid, ~vol.solid
    print(f"  {truth['n_cells']} regions de Voronoi, dont {truth['n_interior_cells']} interieures")
    print(f"  porosite exacte           {truth['porosity']:.4f}")

    section("1. Grandeurs macroscopiques")
    print(f"  porosite mesuree          {ma.metrics.porosity(solid):.4f}")
    frac, _ = ma.metrics.open_porosity(solid)
    print(f"  porosite ouverte          {frac:.4f}")
    sv_bin = ma.metrics.specific_surface(solid)
    print(f"  surface specifique        {sv_bin:.5f} voxel^-1  (sur le masque binaire)")

    section("2. Distance et granulometrie")
    t0 = time.perf_counter()
    dist = ma.distance.distance_transform(fluid)
    aper = ma.granulometry.aperture_map(fluid, n_radii=24)
    psd = ma.granulometry.pore_size_distribution(aper)
    print(f"  distance max              {np.asarray(dist).max():.1f} voxels")
    print(f"  ouverture mediane         {np.median(np.asarray(aper)[fluid]):.1f} voxels")
    mode = psd.loc[psd["fraction"].idxmax(), "size"]
    print(f"  mode de la distribution   {mode:.1f} voxels de diametre")
    print(f"  ({time.perf_counter() - t0:.1f} s)")

    section("3. Marqueurs et segmentation des cellules")
    t0 = time.perf_counter()
    balls = ma.granulometry.maximal_balls(fluid, distance=dist, min_radius=3.0)
    markers = np.asarray(ma.granulometry.cell_markers(fluid, balls=balls, fill_ratio=0.55))
    cells = np.asarray(ma.segmentation.watershed_cells(dist, markers, mask=fluid))
    print(f"  boules maximales          {len(balls)}")
    print(f"  marqueurs retenus         {markers.max()}   (cellules vraies : {truth['n_cells']})")
    print(f"  cellules segmentees       {cells.max()}")

    ref = truth["cell_labels"]
    ious = []
    for lab in truth["interior_cells"]:
        m = ref == lab
        cand, cnt = np.unique(cells[m], return_counts=True)
        best = cand[cnt.argmax()]
        inter = int((m & (cells == best)).sum())
        union = int((m | ((cells == best) & fluid)).sum())
        ious.append(inter / union)
    print(
        f"  IoU sur les interieures   median {np.median(ious):.3f}, "
        f"{100 * np.mean(np.asarray(ious) > 0.7):.0f} % au-dessus de 0,7"
    )
    print(f"  ({time.perf_counter() - t0:.1f} s)")

    section("4. Morphometrie des cellules et des cols")
    cm = ma.segmentation.cell_morphometry(cells)
    th = ma.segmentation.throats(cells)
    cx = ma.segmentation.connectivity(throat_table=th)
    inner = cm[~cm["touches_border"]]
    print(f"  cellules {len(cm)} dont {len(inner)} interieures, cols {len(th)}")
    print(f"  volume total attribue     {100 * cm['volume'].sum() / fluid.sum():.2f} % du fluide")
    if len(inner):
        print(f"  Dpore median              {inner['d_equivalent'].median():.1f} voxels")
        print(
            f"  a/b median                {inner['a_on_b'].median():.2f}   "
            f"(these, Recemat 1723 : 1,302)"
        )
        print(f"  b/c median                {inner['b_on_c'].median():.2f}   (these : 1,232)")
    print(f"  Dcol median               {th['d_equivalent'].median():.1f} voxels")
    print(
        f"  Dcol / Dpore              {th['d_equivalent'].median() / cm['d_equivalent'].median():.2f}"
        f"   (these : 0,53)"
    )
    print(f"  connectivite mediane      {cx.median():.0f}")

    section("5. Forme locale du solide")
    t0 = time.perf_counter()
    st = ma.shape.local_shape_tensor(vol, clip_ratios=False)
    cls = ma.shape.classify_solid(st, solid)
    names = {1: "noeuds", 2: "brins", 3: "plaques"}
    total = int(solid.sum())
    print(f"  {st.n_seeds} points de mesure ({st.params['seeds']})")
    for k, name in names.items():
        print(f"  {name:<24}  {100 * (cls == k).sum() / total:5.1f} % du solide")
    ref_s = truth["strut_mask"] & st.seeds
    ref_n = truth["node_mask"] & st.seeds
    pred = st.a_on_b >= 1.6
    tp, fp = int(pred[ref_s].sum()), int(pred[ref_n].sum())
    print(f"  seuil 1,6 : precision {tp / max(tp + fp, 1):.3f}, rappel {tp / int(ref_s.sum()):.3f}")
    print(f"  ({time.perf_counter() - t0:.1f} s)")

    section("6. Squelette par loi de Plateau")
    t0 = time.perf_counter()
    sk = ma.skeleton.plateau_skeleton(solid, cells)
    deg = (
        sk.edges["node_a"]
        .value_counts()
        .add(sk.edges["node_b"].value_counts(), fill_value=0)
        .reindex(sk.nodes["node"], fill_value=0)
    )
    d_true = ndi.distance_transform_edt(~truth["node_mask"])
    off = np.array(
        [d_true[int(round(r.z)), int(round(r.y)), int(round(r.x))] for r in sk.nodes.itertuples()]
    )
    print(f"  {len(sk.nodes)} noeuds, {len(sk.edges)} brins")
    print(
        f"  cellules par noeud        {100 * (sk.nodes['cells'].map(len) == 4).mean():.0f} % en ont 4"
    )
    print(
        f"  degre des noeuds          mode {int(deg.mode().iloc[0])}, moyenne {deg.mean():.2f}"
        f"   (loi de Plateau : 4)"
    )
    print(f"  longueur de brin mediane  {sk.edges['length'].median():.1f} voxels")
    print(
        f"  ecart au sommet exact     median {np.median(off):.1f} voxel, "
        f"{100 * (off <= 2).mean():.0f} % a moins de 2"
    )
    print(f"  ({time.perf_counter() - t0:.1f} s)")

    print()
    print(f"Total : {time.perf_counter() - t_total:.1f} s")


if __name__ == "__main__":
    main()

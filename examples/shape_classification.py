#!/usr/bin/env python3
"""Classification locale de forme par tenseur d'inertie — demonstration.

Reproduit la figure 3.35 de la these (noeud a~b~c, brin a>>b~c, plaque a~b>>c)
sur des fantomes ou la reponse est connue, puis valide le seuil a/b = 1,6
d'iMorph sur une mousse ou la loi de Plateau est exacte par construction.

    python examples/shape_classification.py

Aucune interface graphique, aucune dependance optionnelle.
"""

from __future__ import annotations

import time

import numpy as np

import morphanalyzer as ma


def canonical_shapes() -> None:
    print("=" * 74)
    print("Les trois formes canoniques (these, fig. 3.35)")
    print("=" * 74)
    print(f"{'fantome':<18}{'a/b':>7}{'b/c':>7}{'a':>7}{'b':>7}{'c':>7}{'elev.':>8}  classe")
    cases = [
        ("cylindres // z", ma.phantoms.cylinders(shape=(64,) * 3, radius=5.0, n=4, axis=0)),
        ("plaque _|_ z", ma.phantoms.plate(shape=(48,) * 3, thickness=6.0, axis=0)),
        ("sphere", ma.phantoms.sphere(shape=(48,) * 3, radius=12.0)),
    ]
    names = {1: "noeud", 2: "brin", 3: "plaque"}
    for label, vol in cases:
        st = ma.shape.local_shape_tensor(vol, clip_ratios=False, propagate=False)
        interior = np.zeros(vol.shape, dtype=bool)
        interior[8:-8, 8:-8, 8:-8] = True
        s = st.seeds & interior
        cls = ma.shape.classify_solid(st, vol.solid)
        dominant = np.bincount(cls[s], minlength=4)[1:].argmax() + 1
        print(
            f"{label:<18}{np.median(st.a_on_b[s]):>7.2f}{np.median(st.b_on_c[s]):>7.2f}"
            f"{np.median(st.a[s]):>7.1f}{np.median(st.b[s]):>7.1f}{np.median(st.c[s]):>7.1f}"
            f"{np.median(st.phi[s]):>8.1f}  {names[dominant]}"
        )


def foam_threshold() -> None:
    print()
    print("=" * 74)
    print("Le seuil a/b = 1,6 sur une mousse (verite terrain = loi de Plateau)")
    print("=" * 74)
    vol = ma.phantoms.voronoi_foam(
        shape=(128,) * 3, n_cells=10, strut=3.0, min_seed_gap=45.0, seed=3
    )
    truth = vol.meta["truth"]
    print(
        f"mousse : porosite {truth['porosity']:.3f}, "
        f"{truth['n_cells']} regions, brins/noeuds exacts connus"
    )

    t0 = time.perf_counter()
    st = ma.shape.local_shape_tensor(vol, clip_ratios=False, propagate=False)
    dt = time.perf_counter() - t0
    print(
        f"tenseur : {st.n_seeds} germes ({st.params['seeds']}), "
        f"{st.n_degenerate} degeneres, {dt:.1f} s"
    )

    struts = truth["strut_mask"] & st.seeds
    nodes = truth["node_mask"] & st.seeds
    for label, m in (("brins", struts), ("noeuds", nodes)):
        q = np.percentile(st.a_on_b[m], [10, 25, 50, 75, 90])
        print(
            f"  a/b {label:<8} p10 {q[0]:.2f}  q1 {q[1]:.2f}  med {q[2]:.2f}  "
            f"q3 {q[3]:.2f}  p90 {q[4]:.2f}   (n={m.sum()})"
        )

    print()
    print(f"{'seuil':>7}{'exactitude':>12}{'rappel':>9}{'precision':>11}")
    for thr in np.arange(1.2, 2.41, 0.2):
        pred = st.a_on_b >= thr
        tp = int(pred[struts].sum())
        fp = int(pred[nodes].sum())
        fn = int((~pred[struts]).sum())
        tn = int((~pred[nodes]).sum())
        flag = "   <- iMorph" if abs(thr - 1.6) < 0.05 else ""
        print(
            f"{thr:>7.1f}{(tp + tn) / (tp + fp + fn + tn):>12.3f}"
            f"{tp / (tp + fn):>9.3f}{tp / max(tp + fp, 1):>11.3f}{flag}"
        )
    print()
    print("Lecture : 1,6 est un choix de haute precision — ce qui est declare brin")
    print("en est un dans ~95 % des cas, au prix d'un quart des brins manques.")
    print("C'est le bon compromis quand on mesure ensuite des diametres et des")
    print("orientations de brins : mieux vaut un echantillon plus petit que")
    print("contamine par des noeuds.")


if __name__ == "__main__":
    canonical_shapes()
    foam_threshold()

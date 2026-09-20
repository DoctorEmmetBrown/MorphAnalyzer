"""Phases 6, 7 et 9 : tortuosite, drainage, percolation d'invasion, os cortical.

Lance tel quel :

    python examples/drainage_cortical.py

Tout tourne sans interface graphique et sans dependance optionnelle.
"""

from __future__ import annotations

import numpy as np

import morphanalyzer as ma


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main() -> None:
    section("1. Mousse de Voronoi : tortuosite geometrique et hydraulique")
    foam = ma.phantoms.voronoi_foam(
        shape=(96, 96, 96), n_cells=20, strut=3.0, min_seed_gap=20.0, seed=7
    )
    fluid = foam.fluid

    tau = ma.tortuosity.plane_tortuosity(fluid, face=0)
    print(f"  tortuosite de plan (z)      : {tau.value:.3f}  (+/- {tau.std:.3f})")

    poi = ma.tortuosity.poiseuille_tortuosity(fluid, n_paths=16)
    print(f"  Poiseuille                  : {poi['tortuosity']:.3f}")
    print(f"  geometrique, memes bouts    : {poi['geometric_tortuosity']:.3f}")
    print(
        f"  distance moyenne a la paroi : {poi['mean_wall_distance_poiseuille']:.2f}"
        f" contre {poi['mean_wall_distance_geometric']:.2f}"
    )

    section("2. Drainage morphologique et courbe de retention")
    for method in ("hazlett", "hilpert"):
        res = ma.network.drainage(fluid, face=0, method=method, step=0.5, surface_tension=0.0728)
        c = res.curve
        half = c.loc[(c["saturation"] - 0.5).abs().idxmin()]
        print(
            f"  {method:8s} : {len(c):3d} points, rayon d'entree {c['radius'].iloc[0]:5.2f}, "
            f"S=50 % a r={half['radius']:5.2f} soit Pc={half['pressure'] / 1e3:7.2f} kPa"
        )

    section("3. Reseau de pores et percolation d'invasion")
    dist = ma.distance.distance_transform(fluid)
    markers = ma.granulometry.cell_markers(fluid, distance=dist, fill_ratio=0.55)
    labels = np.asarray(ma.segmentation.watershed_cells(dist, markers, mask=fluid))
    cells, throats = ma.segmentation.pore_network(labels)
    inlet = ma.network.face_cells(labels, 0)
    outlet = ma.network.face_cells(labels, 1)
    print(f"  {len(cells)} cellules, {len(throats)} cols, {len(inlet)} a l'entree")

    for label, kw in (
        ("cols rigides       ", {}),
        ("avec piegeage      ", {"outlet": outlet, "trapping": True}),
        ("cols deformables x2", {"deformation_rate": 2.0}),
    ):
        r = ma.network.invasion_percolation(cells, throats, inlet=inlet, **kw)
        rmin = np.nanmin(r.cells["filling_radius"])
        print(
            f"  {label} : {r.n_invaded:3d}/{len(cells)} cellules, "
            f"saturation {r.final_saturation:.3f}, dernier rayon {rmin:.2f}"
        )

    section("4. Os cortical : anisotropie angulaire imposee, puis retrouvee")
    bone = ma.phantoms.cortical_tube(
        shape=(24, 128, 128), n_canals=24, sector_weights=[3, 1, 1, 1], seed=2
    )
    t = bone.meta["truth"]
    nz = bone.shape[0]
    canals = np.broadcast_to(t["canal_mask"], (nz,) + t["canal_mask"].shape).copy()
    outside = np.broadcast_to(~t["ring_mask"], canals.shape).copy()

    prof = ma.cortical.angular_profile(canals, center=t["centre"], n_sectors=4, mask_out=outside)
    print(f"  porosite exacte du fantome  : {t['porosity']:.4f}")
    print(f"  porosite mesuree            : {prof.overall:.4f}")
    print("  par secteur (poids 3/1/1/1) :", " ".join(f"{v:.4f}" for v in prof.by_sector()))

    conn = ma.cortical.cortical_connectivity(canals)
    print(
        f"  canaux distincts a la fin   : {conn.table['n_objects'].iloc[-1]}"
        f" (attendu {t['n_canals']})"
    )

    vor, table = ma.cortical.voronoi_2d(np.asarray(bone), mask_out=outside, return_table=True)
    print(f"  territoires de Voronoi 2D   : {table['label'].nunique()} par coupe en moyenne")

    section("5. Maillage et export")
    mesh = ma.mesh.surface_mesh(foam.solid)
    closed = ma.mesh.surface_mesh(foam.solid, pad=True)
    print(f"  surface ouverte : {len(mesh.faces)} triangles, aire {mesh.area:.0f}")
    print(f"  surface fermee  : {len(closed.faces)} triangles, aire {closed.area:.0f}")
    print(f"  volume maille (ferme) : {closed.volume:.0f}, non ferme : {mesh.volume:.0f}")
    print(f"  volume par comptage de voxels : {foam.solid.sum()}")
    print("  -> l'aire se mesure sur la surface ouverte, le volume sur la fermee")


if __name__ == "__main__":
    main()

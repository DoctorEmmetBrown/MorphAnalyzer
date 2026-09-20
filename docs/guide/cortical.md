# Os cortical : profils angulaires et radiaux

Une diaphyse est un tube. Ses propriétés varient surtout avec l'angle autour de
l'axe et avec la distance au centre, d'où le découpage en parts de camembert et
en couronnes.

## Convention d'angle

$$\theta(P) = \operatorname{atan2}(y_0 - j,\; i - x_0)$$

$x$ vers la droite et $y$ vers le **haut** — le sens trigonométrique usuel,
malgré l'axe des lignes qui descend. C'est la convention d'iMorph, conservée
telle quelle. Le secteur 0 couvre donc le quadrant en haut à droite de l'image.

## Porosité par secteur

```python
import morphanalyzer as ma

vol = ma.phantoms.cortical_tube((24, 128, 128), n_canals=20,
                                sector_weights=[3, 1, 1, 1], seed=2)
t = vol.meta["truth"]

prof = ma.cortical.angular_profile(
    canaux, center=t["centre"], n_sectors=12, mask_out=hors_os
)
prof.by_sector()   # moyenne sur toutes les coupes, secteur par secteur
prof.by_slice()    # moyenne sur tous les secteurs, coupe par coupe
prof.pivot()       # table z × secteur, prête à tracer
prof.overall       # porosité totale
```

`mask_out` marque les voxels à **exclure** : ils ne comptent ni au numérateur ni
au dénominateur. C'est ce qu'iMorph appelait le masque.

## Secteurs d'aire égale

Des parts d'angle égal n'ont pas la même aire dès que la section n'est pas
circulaire. Deux corrections :

```python
# section elliptique connue
ma.cortical.sector_bounds(8, ellipse=(a, b, angle_deg))

# forme quelconque : on équilibre le volume réellement présent
ang = ma.cortical.iso_area_angles(interieur_os, 6, center=(y0, x0))
ma.cortical.sector_map(shape, center=(y0, x0), angles=ang)
```

La seconde est l'option « iso apparent volume » d'iMorph : on histogramme le
volume utile par intervalle fin, puis on coupe l'histogramme en parts de somme
égale.

## Ouverture angulaire et profil radial

```python
aper = ma.granulometry.aperture_map(canaux)
ma.cortical.angular_aperture(aper, center=..., n_sectors=12)

ma.cortical.radial_profile(canaux, center=..., n_bins=20, equal_area=True)
```

`equal_area=True` découpe en couronnes de même aire : à épaisseur égale, les
couronnes extérieures contiennent bien plus de pixels, ce qui écrase leur barre
d'erreur par rapport aux intérieures.

## Connectivité par empilement

À la coupe $k$, on ne connaît que le sous-volume $[start, k]$. Deux canaux qui se
rejoignent plus haut sont comptés séparément tant que la jonction n'est pas
atteinte — la mesure dit **à quelle hauteur le réseau se referme sur lui-même**.

```python
prof = ma.cortical.cortical_connectivity(canaux)
prof.table          # z, n_objects, total_volume, largest_fraction, n_merged
prof.volumes[k]     # volumes des objets connus à la coupe k, décroissants
prof.labels         # étiquetage 3D final
prof.n_objects_above(0.05)                  # objets au-delà de 5 % du volume
prof.n_objects_above(0.90, cumulative=True) # combien pour atteindre 90 %
```

Les 853 lignes d'iMorph, avec leur traitement explicite des « ponts » par
intersection et union de `std::set`, deviennent une structure union-find qui les
gère par construction.

## Voronoï 2D

Chaque coupe de la matrice est partagée entre ses pores voisins, par distance
géodésique — donc en contournant les obstacles. Pour de l'os cortical, cela
découpe la matrice en territoires d'ostéones.

```python
labels, table = ma.cortical.voronoi_2d(vol.solid, mask_out=hors_os,
                                       return_table=True)
```

!!! note
    iMorph inondait un relief inversé $\max(d) - d$, c'est-à-dire en partant des
    crêtes. Ici on inonde la distance elle-même depuis les germes de bord, la
    formulation directe du problème — et le même choix que pour le
    [squelette de Plateau](plateau.md), où il avait ramené le décalage des nœuds
    de 1,4 voxel à 0.

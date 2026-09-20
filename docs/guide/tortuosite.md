# Tortuosité

## Ce que mesure une tortuosité

La tortuosité de Carman compare la longueur du plus court chemin dans la phase à
la distance à vol d'oiseau :

$$\tau = \left(\frac{L_{\min}}{\lVert p_1 - p_2 \rVert}\right)^2$$

C'est bien un **carré** — l'usage en milieux poreux, hérité de la loi de
Kozeny–Carman, où $\tau$ intervient au carré dans la perméabilité. Un milieu
libre vaut exactement 1.

La longueur géodésique est obtenue par fast marching : on résout l'équation
eikonale $\lVert \nabla T \rVert = 1/F$ depuis la source, et $T$ est le temps de
parcours minimal.

!!! note "Convention de vitesse"
    iMorph résout $\lVert \nabla T \rVert = F$ : son « champ de vitesse » est en
    réalité une **lenteur**. Le module expose les deux : `speed=` suit la
    convention standard, `cost=` celle d'iMorph.

## Point à point

```python
import morphanalyzer as ma

vol = ma.phantoms.voronoi_foam((128,)*3, n_cells=24, strut=3.0, seed=0)
res = ma.tortuosity.point_tortuosity(vol.fluid, source=(10, 64, 64))

res.value        # tortuosité moyenne sur les points atteints
res.tau          # idem (alias)
res.travel_time  # carte des temps de parcours
res.n_reached    # nombre de voxels atteints
```

## Le piège du premier ordre

Le schéma amont du premier ordre sous-estime les trajets obliques : sur une
diagonale 2D il donne +2,7 %, en 3D +3,8 %. Comme la tortuosité élève ce rapport
au carré, le biais devient +8,8 % — assez pour qu'un milieu libre ressorte à
1,088 au lieu de 1.

D'où le paramètre `reference` :

```python
ma.tortuosity.point_tortuosity(mask, source=..., reference="free_front")
```

`"free_front"` divise le temps de parcours par celui du **même solveur** dans une
boîte vide de même géométrie. Le biais numérique se simplifie et un milieu libre
redonne exactement 1,0000. `"euclidean"` divise par la distance euclidienne,
c'est-à-dire la définition brute, biais compris.

## De face à face, et par direction

```python
ma.tortuosity.plane_tortuosity(mask, face=0)          # z=0 vers z=zmax
ma.tortuosity.directional_tortuosity(mask, angles=16) # rosace dans un pavé inscrit
```

`directional_tortuosity` travaille dans le plus grand pavé inscrit, pour que la
direction mesurée ne soit pas contaminée par la forme de la boîte.

## Tortuosité hydraulique

Un champ de vitesse de Poiseuille pousse le chemin vers l'axe du conduit :

$$F(x) = 1 - \left(1 - \frac{d(x)}{R(x)}\right)^2$$

où $d$ est la distance à la paroi et $R$ l'ouverture locale.

```python
res = ma.tortuosity.poiseuille_tortuosity(mask, n_paths=32)
res["tortuosity"]                      # 1,448 sur une mousse d'essai
res["geometric_tortuosity"]            # 1,403, mêmes extrémités
res["mean_wall_distance_poiseuille"]   # 15,00
res["mean_wall_distance_geometric"]    #  5,17
```

Les deux métriques sont comparées **sur les mêmes extrémités**, faute de quoi la
comparaison ne veut rien dire.

!!! warning "La variante d'iMorph est refusée"
    iMorph utilisait $F = 1 - d^2/R^2$, qui s'annule au **centre** du conduit au
    lieu d'y être maximale. Le résultat dégénère de façon dépendante des données
    ($\tau = 0{,}44$ sur une mousse, 1,02 sur une autre).
    `poiseuille_speed(variant="imorph")` la donne toujours, pour inspection ;
    `poiseuille_tortuosity` la refuse par une `ValueError`.

## Sur le graphe

Quand le réseau de pores est déjà extrait, Dijkstra suffit :

```python
cells, throats = ma.segmentation.pore_network(labels)
tau = ma.tortuosity.graph_tortuosity(cells, throats, axis=0)
```

# Maillage de surface et export

```python
import morphanalyzer as ma

vol = ma.phantoms.sphere((64,)*3, radius=20.0)
mesh = ma.mesh.surface_mesh(vol.solid)

mesh.vertices, mesh.faces, mesh.normals
mesh.area      # aire totale
mesh.volume    # volume enclos
```

## Mailler le masque ou le champ de gris ?

Le masque binaire donne un maillage en escalier, dont l'aire est
systématiquement trop grande. Sur une sphère de rayon 20 :

| entrée | aire mesurée vs analytique |
|---|---:|
| masque binaire | **+9,3 %** |
| champ de distance signée | **−0,08 %** |

Si le champ de gris d'origine — ou une distance signée — est disponible, le
passer :

```python
ma.mesh.surface_mesh(vol.solid, grey=distance_signee, level=0.0)
```

C'est le même constat que pour `metrics.specific_surface`, et la raison pour
laquelle cette fonction mesure sur les niveaux de gris.

## Surface ouverte, volume faux

Quand le solide touche le bord de la boîte — une mousse, un réseau de brins,
presque tout échantillon réel — marching cubes rend une surface **ouverte** : il
ne referme pas les sections coupées par les faces de la boîte.

C'est le bon comportement pour une **aire** : ces sections ne sont pas de
l'interface solide/fluide et n'ont rien à faire dans la surface spécifique. Mais
le **volume** enclos n'a alors aucun sens — sur une mousse de 62 862 voxels
solides, le maillage non fermé donne 6 808.

```python
ma.mesh.surface_mesh(solid)             # aire juste, volume faux
ma.mesh.surface_mesh(solid, pad=True)   # volume juste (61 557), aire majorée
```

`pad=True` entoure le volume d'une couche de fond avant de mailler. À utiliser
pour mesurer un volume, ou pour exporter un objet étanche vers un mailleur ou
une imprimante 3D. Sur un objet qui ne touche pas le bord, les deux donnent
exactement la même chose.

!!! note "Marching cubes biseaute"
    Sur un cube aligné sur la grille, marching cubes remplace les 12 arêtes et
    les 8 coins par des facettes à 45°. Le volume reste juste à 1 % près, l'aire
    non : un chanfrein retire plus d'aire aux faces voisines qu'il n'en ajoute.

## Volume signé

`marching_cubes` oriente les normales vers les valeurs décroissantes du champ,
ce qui donne un volume **négatif** pour un masque. `mesh_volume` rend donc la
valeur absolue par défaut ; `signed=True` pour l'orientation brute.

## Export

STL (binaire et ASCII), OBJ et PLY sont écrits nativement, sans dépendance :

```python
ma.mesh.save_mesh(mesh, "mousse.stl")                # binaire
ma.mesh.save_mesh(mesh, "mousse.stl", binary=False)  # ASCII
ma.mesh.save_mesh(mesh, "mousse.obj")
```

Les autres formats passent par `meshio` (extra `mesh`).

## Décimation

```python
ma.mesh.decimate(mesh, target_reduction=0.5)
```

`trimesh` ou `fast-simplification` (extra `mesh`). La décimation progressive de
Hoppe présente dans les sources d'iMorph (`Thread/Simplificator/progmesh.cpp`,
1 360 lignes) **n'était pas compilée** en 3.2.

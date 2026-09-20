# Drainage, rétention et percolation d'invasion

## Deux algorithmes, une différence de fond

On simule l'intrusion d'un fluide non mouillant par une face, à pression
croissante — donc à rayon de courbure décroissant.

| | critère de passage | référence |
|---|---|---|
| **Hazlett** | un chemin de voxels d'**ouverture locale** $\ge R$ | Hazlett, *Transp. Porous Media* 1995 |
| **Hilpert–Miller** | une boule de rayon $R$ qui **se déplace** continûment | Hilpert & Miller, *Adv. Water Resour.* 2001 |

Hilpert exige en plus que le *centre* de la boule ait un chemin continu : il est
donc toujours plus restrictif ou égal. Sans contrainte d'accès, les deux se
réduisent à l'ouverture morphologique.

```python
import morphanalyzer as ma

res = ma.network.drainage(vol.fluid, face=0, method="hilpert", step=0.5)
res.filling_radius    # carte : plus grand rayon auquel le voxel est atteint
res.curve             # courbe de rétention
```

`filling_radius` vaut `0` pour un voxel poreux jamais envahi et `-1` hors phase.

!!! note "Un écart avec iMorph"
    iMorph initialisait les voxels poreux non envahis à `1.0`, ce qui les faisait
    compter comme envahis au rayon 1 dans la courbe. Corrigé.

## Le blindage par les cols

C'est l'effet « bouteille d'encre » : une chambre large derrière un col étroit
n'est envahie qu'une fois le rayon descendu **sous celui du col**, quelle que
soit sa propre taille. C'est la seule chose qu'un algorithme de drainage
morphologique doit absolument reproduire, et c'est ce que vérifie
`test_ink_bottle_shielding`.

## Pression capillaire

```python
ma.network.capillary_pressure(10.0, unit="um")   # 14 560 Pa (eau/air, θ=0)
```

Par défaut, Young–Laplace : $P_c = 2\sigma\cos\theta / r$.

!!! warning "La convention d'iMorph vaut le double"
    iMorph écrivait $P_c = 4\sigma / r$ avec $r$ la colonne intitulée
    « radius ball ». Cela revient à lire l'ouverture comme un **diamètre**. Le
    facteur 2 est systématique — il ne change pas la forme de la courbe, mais il
    décale l'axe des pressions. Accessible par `convention="imorph"`.

## Percolation d'invasion

Sur le réseau cellules–cols, l'amas envahisseur franchit à chaque étape le plus
grand col accessible.

```python
cells, throats = ma.segmentation.pore_network(labels)
inlet  = ma.network.face_cells(labels, 0)
outlet = ma.network.face_cells(labels, 1)

res = ma.network.invasion_percolation(cells, throats, inlet=inlet,
                                      outlet=outlet, trapping=True)
res.cells["filling_radius"]   # escalier décroissant
res.curve                     # saturation cumulée
```

L'entrée dans une cellule de la face d'injection est commandée par le rayon de
la **cellule** ; le passage d'une cellule à l'autre, par celui du **col**.

La boucle en pression d'iMorph (`r_current -= 0.2`) est remplacée par un tas
binaire, qui donne exactement le même ordre d'invasion sans balayer une grille
de pressions.

### Cols déformables

Une trouvaille d'iMorph pour les milieux élastiques : les cols s'élargissent avec
la pression. Le facteur $k$ vaut 1 à la pression la plus basse et
`deformation_rate` à la plus haute ; comme $P_c \propto 1/r$, on a
$k(r) = A/r + B$, et un col de rayon $R$ passe dès que $k(r)R \ge r$ :

$$r_{\text{eff}} = \frac{BR + \sqrt{B^2R^2 + 4AR}}{2}, \qquad
A = \frac{D-1}{2 - 1/r_{\max}}, \quad B = D - 2A$$

Calculable une fois pour toutes. Ni $\sigma$ ni la taille de voxel n'y
interviennent — elles se simplifient.

!!! warning
    Le rayon de référence 0,5 est en **voxels**. Travailler en voxels si l'on
    utilise `deformation_rate`.

## Export

```python
ma.network.to_openpnm(cells, throats, length_scale=1e-6)   # extra `network`
ma.network.save_network_text(cells, throats, "reseau.txt") # format iMorph
```

# wm_colab

Prototype expérimental d’un modèle latent inspiré de [LeWorldModel](https://arxiv.org/html/2603.19312v2), entraîné sur un gridworld isométrique 5×5.

Ce projet explore si un modèle peut apprendre, à partir de pixels et d’actions uniquement, une représentation latente structurée suffisante pour :
- encoder la position de l’agent,
- prédire des transitions locales,
- et supporter une forme simple de planification.

L’objectif n’est pas de démontrer un “world model” robuste au sens fort, mais d’étudier concrètement ce qu’un tel système apprend dans un environnement minimal.

---

## Résultats

| Métrique | Score |
|---|---:|
| Sonde linéaire — colonne (R²) | 1.000 |
| Sonde linéaire — ligne (R²) | 1.000 |
| Planification MPC — 600 paires départ→but | 600 / 600 (100%) |

Dans ce cadre simple, la planification MPC fonctionne pour toutes les distances de Manhattan (1 à 8), avec un nombre de pas optimal.

Ces résultats doivent être interprétés dans le contexte d’un environnement très simple, entièrement observable, déterministe, et évalué sur la même distribution que celle utilisée à l’entraînement.

---

## Ce que ce projet montre

- mise en place d’un protocole d’apprentissage auto-supervisé cohérent ;
- apprentissage d’une représentation latente fortement structurée ;
- encodage de la position de l’agent, linéairement décodable (R² = 1.0) ;
- analyse de failure modes :
  - shortcut sur la cible dans une version antérieure,
  - collapse pour une régularisation insuffisante,
  - caractère trompeur de la distance L2 brute dans l’espace latent ;
- utilisation downstream du latent pour une planification simple (MPC).

---

## Ce que ce projet ne montre pas

- pas de preuve de généralisation hors distribution ;
- pas de robustesse à une augmentation forte de la complexité ;
- pas de planification ouverte à long horizon ;
- pas d’environnement partiellement observable ;
- pas de dynamique visuelle riche ou bruitée ;
- pas de validation scientifique forte (benchmarks, multi-seeds, ablations complètes).

Autrement dit : le projet montre surtout un apprentissage de géométrie d’état exploitable, pas encore un “world modeling” fort.

---

## Architecture

- **Encodeur** : ViT-Tiny  
  `patch_size=16`, `embed_dim=192`, `depth=4`, `heads=3`, CLS token
- **Prédicteur** : Transformer (2 blocs) avec AdaLN, conditionné par l’action
- **Régularisation** : SIGReg  
  `M=1024`, `λ=10.0`
- **Loss** : MSE de prédiction + `λ · SIGReg`, entraînement end-to-end, sans stop-gradient
- **Nombre de paramètres** : 3.03M

---

## Environnement

- gridworld isométrique 5×5 ;
- agent mobile en 4 directions ;
- observations visuelles en niveaux de gris (128×128) ;
- transitions locales (T = 2) ;
- dataset de 50k transitions.

---

## Structure du projet

```text
wm_colab/
├── src/
│   ├── env.py                  # GridWorld 5×5 + rendu isométrique
│   └── model.py                # modèle latent (ViT + Transformer + SIGReg)
├── notebooks/
│   ├── 01_wm_dataset.ipynb     # génération du dataset (50k transitions)
│   ├── 02_wm_train.ipynb       # entraînement (200 epochs, Colab GPU)
│   ├── 03_wm_probe.ipynb       # sondes linéaires + PCA
│   └── 04_wm_plan.ipynb        # planification MPC à partir du latent
├── saved_models/
│   └── lewm_vit_50k_v5.pt      # checkpoint final
└── requirements.txt
````

---

## Workflow

Développement local → versioning Git → entraînement sur Google Colab (GPU) → analyse dans notebooks

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Remarques

Ce dépôt documente un prototype exploratoire centré sur l’apprentissage de représentations.

Il met l’accent sur :

* la construction d’un pipeline auto-supervisé fonctionnel ;
* l’analyse de ce que le modèle encode réellement (plutôt que la performance brute) ;
* l’identification de shortcuts et de collapse ;
* la transformation d’un latent en usage downstream simple.

---

## Pistes d’amélioration

* entraîner sur des séquences de longueur `T > 2` ;
* évaluer hors distribution (nouvelles configurations, tailles de grille) ;
* ajouter obstacles / objets interactifs ;
* introduire du bruit ou des variations visuelles ;
* utiliser une planification plus robuste (ex: CEM) ;
* comparer avec des baselines plus simples (CNN, modèles non latents).

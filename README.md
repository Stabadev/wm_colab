# wm_colab

Implémentation d'un World Model latent de type [LeWorldModel](https://arxiv.org/abs/2406.14277) sur un gridworld isométrique 5×5.

Le modèle apprend une représentation latente des observations visuelles et prédit les états futurs conditionnellement aux actions, sans supervision explicite de la position.

## Résultats

| Métrique | Score |
|---|---|
| Sonde linéaire col (R²) | 1.000 |
| Sonde linéaire row (R²) | 1.000 |
| Planification MPC — 600 paires | **600/600 (100%)** |

La planification MPC fonctionne à toutes les distances Manhattan (1 à 8), avec un nombre de pas optimal à chaque fois.

## Architecture

- **Encodeur** : ViT-Tiny (patch_size=16, D=192, 4 blocs, 3 têtes, CLS token)
- **Prédicteur** : Transformer 2 blocs AdaLN, conditionné par action
- **Régularisation** : SIGReg (M=1024 projections, λ=10.0) — maintient les embeddings proches de N(0,1)
- **Loss** : MSE prédiction + λ·SIGReg, end-to-end, sans stop-gradient
- **Paramètres** : 3.03M

## Structure

```
wm_colab/
├── src/
│   ├── env.py          — GridWorld 5×5 + rendu isométrique
│   └── model.py        — LeWorldModel (ViT + AdaLN Transformer + SIGReg)
├── notebooks/
│   ├── 01_wm_dataset.ipynb   — génération du dataset (50k transitions)
│   ├── 02_wm_train.ipynb     — entraînement (200 epochs, Colab G4)
│   ├── 03_wm_probe.ipynb     — sondes linéaires + visualisation PCA
│   └── 04_wm_plan.ipynb      — planification MPC dans l'espace latent
├── saved_models/
│   └── lewm_vit_50k_v5.pt    — checkpoint final
└── requirements.txt
```

## Workflow

Développement local → Git → Google Colab (entraînement GPU) → résultats sauvegardés dans les notebooks.

## Installation

```bash
pip install -r requirements.txt
```

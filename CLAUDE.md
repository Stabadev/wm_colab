# CLAUDE.md — Instructions pour Claude Code

## Contexte projet

World Model latent sur gridworld isométrique 5×5, inspiré de LeWorldModel (LeWM).
Repo : https://github.com/Stabadev/wm_colab

## Règle fondamentale

Tout le code s'écrit en local. Jamais de modification dans Colab.
Cycle : LOCAL → GIT → COLAB → DRIVE → LOCAL

## Structure

```
src/
  env.py     — GridWorld + rendu isométrique (complet, ne pas modifier sans raison)
  model.py   — LeWorldModel (ViT encoder + Transformer AdaLN + SIGReg)
wm_dataset.ipynb  — génération du dataset (complet)
wm_train.ipynb    — entraînement LeWM (complet)
test_colab.ipynb  — validation workflow (complet)
local_runs/       — sorties locales (gitignored)
```

Règle : logique métier dans `src/`, notebooks = orchestration uniquement.

## Workflow hybride local/Colab

Chaque notebook détecte l'environnement avec :
```python
IN_COLAB = "google.colab" in sys.modules
```

- Local : chemins `./local_runs/`
- Colab : monte Drive, clone le repo, chemins Drive

Sur Colab, le dataset doit être présent sur Drive avant l'entraînement
(généré par `wm_dataset.ipynb` lancé sur Colab, ou uploadé manuellement).

## Conventions de code

- Français pour les commentaires et les prints
- Constantes d'hyperparamètres en MAJUSCULES dans les notebooks
- `src/model.py` : type hints sur toutes les signatures publiques
- Shapes commentées sur les lignes clés : `# (B, T, D)`
- `register_buffer` pour tout tenseur fixe (non entraînable) qui doit suivre `.to(device)`

## Architecture LeWM (état actuel)

- **Encodeur** : ViT-Tiny (patch_size=16, D=192, 4 blocs, 3 têtes, CLS token)
- **Prédicteur** : Transformer 2 blocs AdaLN, conditionné par action via embedding
- **SIGReg** : M=1024 projections aléatoires fixes, pénalise écart à N(0,1), λ=0.1
- **Loss** : MSE prédiction + λ·SIGReg, end-to-end, sans stop-gradient
- **Entrées** : obs `(B, T, H, W)` float [0,1] + actions `(B,)` int64
- **T=2** avec le dataset actuel (transitions individuelles)

## Dataset

Format `.npz` : `obs_t`, `actions`, `obs_t1`, `agent_t`, `agent_t1`, `target`
Images : 128×128 niveaux de gris uint8
Normalisation dans le Dataset PyTorch (÷ 255.0), pas dans le modèle.

## Points d'attention

- `elementwise_affine=False` obligatoire sur les LayerNorm du prédicteur (les LN sont remplacées par AdaLN)
- Les frames sont encodées **séparément** : reshape `(B*T, 1, H, W)` avant l'encodeur
- Le collapse se surveille via `emb.std()` — si std < 0.05 : augmenter `sigreg_lambda`
- `embed_dim` doit être pair (PE sinusoïdale dans le prédicteur)

## Prochaines étapes

- [ ] Générer un dataset plus grand sur Colab (50k+ transitions)
- [ ] Évaluation : visualisation des embeddings (UMAP/t-SNE)
- [ ] Planification MPC dans l'espace latent

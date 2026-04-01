# PLAN_COLAB.md

## Contexte du projet

Ce projet implémente un World Model inspiré de LeWorldModel (LeWM) sur un environnement gridworld isométrique simple.

Objectif : construire un pipeline complet reproductible :
1. Environnement (`src/env.py`)
2. Dataset (`wm_dataset.ipynb`)
3. Training (`wm_train.ipynb`)

Le workflow est hybride :
- développement en local
- exécution sur Google Colab
- stockage sur Google Drive
- versionning GitHub

---

## Structure du repo

wm_colab/
├── src/
│   ├── env.py
│   └── model.py
├── test_colab.ipynb
├── wm_dataset.ipynb
├── wm_train.ipynb
├── local_runs/
│   ├── datasets/
│   └── checkpoints/
├── local_test_run/
├── README.md
├── requirements.txt
└── .gitignore

Règle :
- logique métier dans src/
- notebooks = orchestration uniquement

---

## Workflow

### 1. Local
- coder / tester
- petits datasets

### 2. Git
git add
git commit
git push

Repo :
https://github.com/Stabadev/wm_colab

### 3. Colab

Dans notebook :

git clone https://github.com/Stabadev/wm_colab.git
cd /content/wm_colab
git pull

Ajouter au path :
import sys
sys.path.insert(0, "/content/wm_colab")

### 4. Drive

Mon Drive/projetColab/wm_colab/
├── datasets/
├── checkpoints/
└── test_run/

Règle :
- Colab = stateless
- tout sauvegarder sur Drive

---

## Notebooks hybrides

Chaque notebook fonctionne :

- en local
- en Colab

Détection :

IN_COLAB = "google.colab" in sys.modules

Local :
- ./local_runs/

Colab :
- montage Drive
- clone repo
- stockage Drive

---

## Dataset

Format .npz :
- obs_t
- actions
- obs_t1
- agent_t
- agent_t1
- target

---

## Modèle

Étape 1 :
- CNN + MLP
- SIGReg simple

Étape 2 :
- ViT encoder
- Transformer predictor
- AdaLN
- SIGReg fidèle

---

## Ordre de dev

- [x] workflow
- [x] env
- [x] dataset
- [x] training (src/model.py + wm_train.ipynb)
- [ ] LeWM complet (évaluation, planification MPC)

---

## Règle fondamentale

❌ pas de modif dans Colab
✅ toujours local → git → Colab

Cycle :
LOCAL → GIT → COLAB → DRIVE → LOCAL

---

## Architecture implémentée (src/model.py)

### Vue d'ensemble

```
obs (B, T, H, W)
        │
        ▼
┌───────────────┐
│  ViTEncoder   │  × T frames encodées indépendamment
│  (4 blocs)    │
└───────────────┘
        │ emb (B, T, D=192)
        ▼
┌───────────────────────────┐
│  TransformerPredictor     │  conditionné par action via AdaLN
│  (2 blocs AdaLN)          │
└───────────────────────────┘
        │ pred (B, T, D)
        ▼
┌──────────────────────────────────────────┐
│  Loss = MSE(pred[:,:-1], emb[:,1:])      │
│       + λ · SIGReg(emb)                 │
└──────────────────────────────────────────┘
```

**3.03M paramètres au total.**

---

### Encodeur ViT

L'encodeur transforme une image 128×128 en un vecteur de dimension D=192.

**Étape 1 — Patch Embedding**

L'image est découpée en patches de 16×16 pixels.
Pour une image 128×128 : 8×8 = **64 patches**.

Chaque patch est aplati (256 valeurs pour un patch 16×16 en niveaux de gris) et projeté dans un espace de dimension D=192 via une convolution :

```
Conv2d(in_channels=1, out_channels=192, kernel_size=16, stride=16)
image (1, 128, 128) → 64 vecteurs de dim 192
```

**Étape 2 — CLS token + position embedding**

Un token spécial apprenable (`cls_token`) est ajouté en tête de la séquence.
La séquence passe de 64 à 65 tokens.

Un embedding de position (aussi appris) est ajouté à chaque token pour lui donner conscience de sa position spatiale dans l'image.

**Étape 3 — 4 blocs Transformer (EncoderBlock)**

Chaque bloc applique :
1. LayerNorm → Self-Attention (tous les tokens s'observent mutuellement)
2. LayerNorm → MLP (deux couches linéaires avec GELU)

Avec des connexions résiduelles : `x = x + attention(x)` puis `x = x + mlp(x)`.

La self-attention a 3 têtes (head_dim = 192/3 = 64 par tête).

**Étape 4 — Extraction du CLS token**

Après la LayerNorm finale, seul le CLS token (position 0) est gardé.
Il concentre l'information globale de l'image : `(B, D=192)`.

**Traitement des séquences temporelles**

Pour T=2 frames, les deux images sont encodées **séparément** (pas ensemble dans le même Transformer). On reshapes `(B, T, H, W)` → `(B*T, 1, H, W)`, on encode, puis on reshape en `(B, T, D)`.

---

### Prédicteur Transformer avec AdaLN

Le prédicteur reçoit la séquence de latents `emb (B, T, D)` et l'action `(B,)`, et produit les prédictions `pred (B, T, D)`.

**Position embedding sinusoïdale**

Contrairement à l'encodeur (position apprise), le prédicteur utilise un embedding sinusoïdal fixe. Il encode la position temporelle de chaque latent dans la séquence.

**AdaLN — Adaptive Layer Normalization**

C'est le mécanisme clé du conditionnement par l'action.

Dans un Transformer standard, la LayerNorm applique :
```
LN(x) = (x - mean) / std * γ + β
```
où γ et β sont des paramètres **fixes** appris.

Dans AdaLN, γ et β sont **générés dynamiquement** à partir de l'action :

```
action (B,)
  → Embedding → action_emb (B, D)
  → SiLU + Linear(D, 4*D)
  → split : scale1, shift1, scale2, shift2  — chacun (B, D)

AdaLN(x) = x * (1 + scale) + shift
```

Ainsi, pour chaque action différente, la normalisation est différente. Le modèle apprend comment l'action doit moduler le traitement des latents.

Les LayerNorm dans les blocs du prédicteur n'ont **pas** de paramètres propres (`elementwise_affine=False`) : l'AdaLN s'en charge entièrement.

**Masque causal**

La self-attention dans le prédicteur est causale : la position t ne peut "voir" que les positions 0..t. Cela garantit que la prédiction à t n'utilise pas d'information du futur. Pour T=2, c'est trivial mais le code est correct pour T>2.

**2 blocs AdaLNBlock**

Structure de chaque bloc :
1. AdaLN (norm1) → Self-Attention causale
2. AdaLN (norm2) → MLP

---

### SIGReg — Régularisation anti-collapse

**Le problème du collapse**

Sans régularisation, la solution triviale est que l'encodeur mappe tout vers le même vecteur. La loss MSE serait 0 mais le modèle n'aurait rien appris.

**La solution : SIGReg**

SIGReg force les embeddings à se comporter comme une gaussienne isotrope N(0, I) dans toutes les directions.

**Comment ça marche :**

1. On tire M=1024 directions aléatoires unitaires `r_m ∈ R^D` (fixes, non entraînables)
2. On projette les embeddings du batch sur chacune : `h_m = Z @ r_m` → scalaires
3. On pénalise tout écart à N(0, 1) sur ces projections :

```
pour chaque direction m :
    penalite_mean = mean(h_m)²        # force la moyenne à 0
    penalite_var  = (mean(h_m²) - 1)² # force la variance à 1

SIGReg = moyenne sur les M directions
```

La matrice de projection est stockée comme un `register_buffer` PyTorch : elle est fixe (non entraînée), mais se déplace automatiquement sur GPU avec `.to(device)`.

**SIGReg est appliqué sur chaque pas de temps** et moyenné : avec T=2, on régularise les embeddings de obs_t et obs_t1 séparément.

---

### Loss complète

```
L = L_pred + λ · L_SIGReg

L_pred   = MSE(pred[:, :-1], emb[:, 1:])
           = MSE entre la prédiction du prédicteur à t
             et l'embedding réel à t+1

L_SIGReg = moyenne SIGReg sur chaque timestep

λ = 0.1
```

**Sans stop-gradient** : le gradient remonte à la fois dans le prédicteur ET dans l'encodeur via `emb[:, 1:]`. C'est la SIGReg qui empêche le collapse, pas un stop-gradient.

---

### Données d'entraînement (T=2)

Le dataset contient des transitions individuelles `(obs_t, action, obs_t1)`.

Pour LeWM, on forme des séquences de longueur T=2 :

```
obs     = stack([obs_t, obs_t1])  → (B, 2, H, W) float [0,1]
actions = action                  → (B,) int64
```

Le modèle encode les deux frames → `emb (B, 2, D)`,
prédit depuis la première → `pred[:, 0]`,
compare avec l'embedding de la deuxième → `emb[:, 1]`.

---

### Hyperparamètres par défaut

| Paramètre | Valeur | Raison |
|---|---|---|
| patch_size | 16 | 64 patches/image, bon compromis résolution/coût |
| embed_dim D | 192 | ViT-Tiny, adapté au gridworld simple |
| n_heads | 3 | head_dim=64, standard |
| encoder_depth | 4 | capacité suffisante pour 5×5 |
| predictor_depth | 2 | prédiction simple (1 pas) |
| sigreg_M | 1024 | estimation robuste de la distribution |
| sigreg_lambda | 0.1 | équilibre pred/regularisation |
| batch_size | 64 | adapté GPU Colab T4 |
| lr | 3e-4 | Adam standard |

---

## Modèle à implémenter

## LeWorldModel (LeWM)

LeWorldModel (LeWM) est un modèle de monde latent appris dans un cadre **offline, sans récompense**, à partir de trajectoires d’observations et d’actions. L’objectif est d’apprendre une représentation latente des dynamiques de l’environnement, indépendante de toute tâche, qui pourra ensuite être exploitée pour la planification et la prise de décision.

### Apprentissage du modèle de monde latent

Le modèle est entraîné sur un dataset offline composé de trajectoires de longueur (T), contenant des observations visuelles (o_{1:T}) (pixels) et des actions associées (a_{1:T}). Aucune récompense ni signal de supervision explicite n’est utilisé. Les données peuvent provenir de politiques exploratoires ou pseudo-expertes, tant qu’elles couvrent suffisamment les dynamiques de l’environnement.

Le modèle est composé de deux éléments principaux :

* **Encoder** : transforme une observation (o_t) en une représentation latente compacte (z_t)
* **Predictor** : prédit le prochain état latent (\hat{z}_{t+1}) à partir de (z_t) et de l’action (a_t)

[
z_t = enc_\theta(o_t)
\quad,\quad
\hat{z}*{t+1} = pred*\phi(z_t, a_t)
]

L’encodeur est implémenté comme un Vision Transformer (ViT), et le prédicteur comme un Transformer conditionné par les actions via Adaptive Layer Normalization (AdaLN). Le modèle est entraîné de manière auto-régressive sur des séquences temporelles.

### Fonction de coût

L’apprentissage repose sur deux termes :

1. **Loss de prédiction** :
   [
   \mathcal{L}*{pred} = | \hat{z}*{t+1} - z_{t+1} |_2^2
   ]

2. **Régularisation anti-collapse (SIGReg)** :
   Cette régularisation force les embeddings à suivre une distribution gaussienne isotrope en projetant les représentations sur des directions aléatoires et en appliquant un test statistique univarié.

[
\text{SIGReg}(Z) = \frac{1}{M} \sum_{m=1}^{M} T(h^{(m)})
]

La loss complète est :

[
\mathcal{L}*{LeWM} = \mathcal{L}*{pred} + \lambda \cdot \text{SIGReg}(Z)
]

avec (M = 1024) projections et (\lambda = 0.1).

Le modèle est entraîné **end-to-end**, sans stop-gradient, EMA ou heuristiques supplémentaires.

### Pseudo-code

```python
def LeWorldModel(obs, actions, lambd=0.1):
    emb = encoder(obs)                 # (B, T, D)
    next_emb = predictor(emb, actions) # (B, T, D)

    pred_loss = F.mse_loss(emb[:, 1:], next_emb[:, :-1])
    sigreg_loss = mean(SIGReg(emb.transpose(0, 1)))

    return pred_loss + lambd * sigreg_loss
```

### Intuition

Le modèle apprend à **représenter le monde dans un espace latent prédictif**, où l’évolution des états est plus simple à modéliser. La régularisation SIGReg empêche les solutions triviales (collapse) et garantit une représentation riche et exploitable pour la planification future (ex: MPC).


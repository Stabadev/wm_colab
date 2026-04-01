# LeWorldModel ISO — Document de travail complet

> **Pour Claude** : ce fichier est autonome. Tout le code de référence est ici.
> L'objectif est de construire un repo propre avec un notebook Colab-ready.
> On travaille ensemble en local (JupyterLab) et on exécute sur Google Colab.
> Date de départ : 2026-03-31

---

## Contexte du projet

On implémente **LeWorldModel (LeWM)** sur un environnement gridworld iso simple.
Référence : arxiv 2603.19312.

L'idée centrale : apprendre un world model en espace latent.
- Un **encoder** compresse les observations pixel → vecteur latent z
- Un **predictor** prédit le prochain latent z_t+1 depuis (z_t, action)
- Pas de stop-gradient, pas d'EMA, gradients partout (end-to-end)
- Anti-collapse via **SIGReg** (Sketched Isotropic Gaussian Regularizer)

L'environnement : board N×N (défaut 5×5), rendu isométrique PIL, niveaux de gris.

---

## 1. Structure cible du repo

```
lewm-iso/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── env.py          # GridWorld + rendu ISO (tout le code rendu est ici)
│   └── model.py        # Encoder CNN + Predictor + SIGReg + loss
└── lewm_dataset.ipynb  # Notebook dataset (Colab-ready)
└── lewm_train.ipynb    # Notebook entraînement (Colab-ready)
```

**Règle absolue** : les notebooks ne contiennent pas de logique métier.
Tout est dans `src/`. Les notebooks font `from src.env import ...`.

---

## 2. Conventions importantes

### Système de coordonnées du board
```
col = axe horizontal  (0 = gauche, N-1 = droite)
row = axe vertical    (0 = haut,   N-1 = bas)
position = (col, row)
```

### Actions
```python
DELTAS = {
    0: ( 0, +1),   # haut   (row augmente)
    1: ( 0, -1),   # bas    (row diminue)
    2: (-1,  0),   # gauche (col diminue)
    3: (+1,  0),   # droite (col augmente)
}
N_ACTIONS = 4
ACTION_NAMES = {0:'↑ haut', 1:'↓ bas', 2:'← gauche', 3:'→ droite'}
```
Collision bord : si l'action sort du board, l'agent reste en place (silencieuse).

### Observations
- Niveaux de gris, `uint8`, shape `(RENDER_SIZE, RENDER_SIZE)` = `(128, 128)`
- Le board est toujours visible en entier (projection iso auto-fitée)
- L'agent est un carré gris clair centré dans sa case (avec padding PAD_AGENT)
- La cible est un carré gris foncé qui remplit toute sa case
- Quand l'agent est sur la cible, les deux sont visibles (l'agent est plus petit)

### Dataset
Chaque transition stockée : `(obs_t, action, obs_t1, agent_t, agent_t1, target)`
- `obs_t`, `obs_t1` : images uint8 (pour entraînement)
- `agent_t`, `agent_t1`, `target` : positions (col, row) int8 (pour probing/stats)

---

## 3. Code source complet — `src/env.py`

```python
"""
env.py — GridWorld isométrique pour LeWorldModel
=================================================
Contient :
  - Fonctions de projection 3D → 2D (caméra iso)
  - precalculate_render(cfg) → RenderCache  (à appeler une seule fois)
  - render_obs(agent_pos, target_pos, cache) → np.ndarray uint8
  - class GridWorld
"""

import numpy as np
from PIL import Image, ImageDraw
from dataclasses import dataclass


# ── Config rendu ──────────────────────────────────────────────────────────────

@dataclass
class RenderConfig:
    # Board
    N: int = 5              # taille N×N

    # Caméra
    az_deg: float = 203     # azimut (rotation autour du board)
    el_deg: float = 32      # élévation (hauteur caméra)
    margin_pct: float = 1   # marge autour du board (% de RENDER_SIZE)

    # Rendu
    render_size: int = 128  # résolution de sortie en pixels

    # Couleurs / niveaux de gris
    color_bg: tuple = (255, 255, 255)    # fond image : blanc
    color_board: tuple = (210, 235, 228) # fond board : vert-gris
    color_line: tuple = (17, 17, 17)     # lignes grille : noir
    gray_agent: int = 141                # agent : gris clair
    gray_target: int = 60               # cible : gris foncé
    pad_agent: float = 0.32             # padding agent (0 = toute la case)
    line_w: float = 1.2                 # épaisseur lignes (px à 128px ref)


# ── Maths projection ──────────────────────────────────────────────────────────

def _cross(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ], dtype=float)

def _norm(v):
    l = np.linalg.norm(v)
    return v / l if l > 1e-9 else v

def _build_camera(az_deg, el_deg, N):
    cx, cz = N / 2, N / 2
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    cam_dir = np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])
    forward = _norm(-cam_dir)
    rv      = _norm(_cross(forward, np.array([0., 1., 0.])))
    uv      = _cross(rv, forward)
    return dict(cx=cx, cz=cz, cam_dir=cam_dir, forward=forward, rv=rv, uv=uv)

def _project_vertex(col, row, cam, dist, focal, size):
    cam_pos = np.array([cam['cx'], 0., cam['cz']]) + cam['cam_dir'] * dist
    d = np.array([col, 0., row], dtype=float) - cam_pos
    px = np.dot(d, cam['rv'])
    py = np.dot(d, cam['uv'])
    pz = np.dot(d, cam['forward'])
    if pz <= 0.01:
        return None
    return np.array([
        size / 2 + focal * px / pz * size,
        size / 2 - focal * py / pz * size,
    ])

def _auto_fit(cam, N, size, margin_pct):
    """Trouve automatiquement dist et focal pour que le board tienne dans l'image."""
    margin = size * margin_pct / 100
    target = size - 2 * margin
    for focal in np.arange(2.5, 0.4, -0.1):
        for dist in np.arange(1.0, 30.0, 0.2):
            pts, ok = [], True
            for row in range(N + 1):
                for col in range(N + 1):
                    p = _project_vertex(col, row, cam, dist, focal, size)
                    if p is None:
                        ok = False; break
                    pts.append(p)
                if not ok:
                    break
            if not ok:
                continue
            pts = np.array(pts)
            minx, miny = pts[:, 0].min(), pts[:, 1].min()
            maxx, maxy = pts[:, 0].max(), pts[:, 1].max()
            w, h = maxx - minx, maxy - miny
            if w <= target and h <= target and w > target * 0.6 and h > target * 0.6:
                return dist, focal, size / 2 - (minx + maxx) / 2, size / 2 - (miny + maxy) / 2
    return 5.0, 1.5, 0., 0.

def _lerp2(a, b, t):
    return a + (b - a) * t


# ── Cache de rendu ─────────────────────────────────────────────────────────────

class RenderCache:
    """Précalcule tous les quads et lignes une seule fois."""
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        N    = cfg.N
        size = cfg.render_size

        cam = _build_camera(cfg.az_deg, cfg.el_deg, N)
        dist, focal, ox, oy = _auto_fit(cam, N, size, cfg.margin_pct)

        # Projeter tous les sommets
        verts = {}
        for row in range(N + 1):
            for col in range(N + 1):
                p = _project_vertex(col, row, cam, dist, focal, size)
                if p is not None:
                    verts[(col, row)] = p + np.array([ox, oy])

        def make_quad(col, row, pad):
            corners = [
                verts.get((col,   row)),
                verts.get((col+1, row)),
                verts.get((col+1, row+1)),
                verts.get((col,   row+1)),
            ]
            if any(p is None for p in corners):
                return None
            center = sum(corners) / 4
            return [
                (float(_lerp2(p, center, pad)[0]), float(_lerp2(p, center, pad)[1]))
                for p in corners
            ]

        self.board_quads = {}
        self.agent_quads = {}
        for row in range(N):
            for col in range(N):
                q = make_quad(col, row, 0)
                if q:
                    self.board_quads[(col, row)] = q
                q = make_quad(col, row, cfg.pad_agent)
                if q:
                    self.agent_quads[(col, row)] = q

        self.grid_lines = []
        for row in range(N + 1):
            pts = [
                (float(verts[(col, row)][0]), float(verts[(col, row)][1]))
                for col in range(N + 1) if (col, row) in verts
            ]
            if len(pts) >= 2:
                self.grid_lines.append(pts)
        for col in range(N + 1):
            pts = [
                (float(verts[(col, row)][0]), float(verts[(col, row)][1]))
                for row in range(N + 1) if (col, row) in verts
            ]
            if len(pts) >= 2:
                self.grid_lines.append(pts)

        # Niveaux de gris précalculés
        c = cfg.color_board
        self.gray_board = int(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
        c = cfg.color_line
        self.gray_line = int(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
        self.lw = max(1, round(cfg.line_w * size / 128))


def precalculate_render(cfg: RenderConfig = None) -> RenderCache:
    """Appeler une seule fois avant de générer le dataset."""
    if cfg is None:
        cfg = RenderConfig()
    return RenderCache(cfg)


# ── Rendu ─────────────────────────────────────────────────────────────────────

def render_obs(agent_pos, target_pos, cache: RenderCache) -> np.ndarray:
    """
    Rend une observation isométrique.

    agent_pos, target_pos : (col, row) dans [0, N-1]
    cache                 : RenderCache précalculé

    Retourne : np.ndarray (H, W) uint8, niveaux de gris
    """
    cfg  = cache.cfg
    size = cfg.render_size

    img  = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)

    # Fond du board
    for q in cache.board_quads.values():
        draw.polygon(q, fill=cache.gray_board)

    # Cible (dessinée avant l'agent — reste visible grâce au padding)
    tc, tr = target_pos
    if (tc, tr) in cache.board_quads:
        draw.polygon(cache.board_quads[(tc, tr)], fill=cfg.gray_target)

    # Agent (avec padding)
    ac, ar = agent_pos
    if (ac, ar) in cache.agent_quads:
        draw.polygon(cache.agent_quads[(ac, ar)], fill=cfg.gray_agent)

    # Lignes de grille
    for pts in cache.grid_lines:
        draw.line(pts, fill=cache.gray_line, width=cache.lw)

    return np.array(img, dtype=np.uint8)


# ── GridWorld ─────────────────────────────────────────────────────────────────

DELTAS = {
    0: ( 0, +1),   # haut
    1: ( 0, -1),   # bas
    2: (-1,  0),   # gauche
    3: (+1,  0),   # droite
}

class GridWorld:
    """
    Environnement gridworld N×N.
    Position = (col, row), col=0 gauche, row=0 haut.
    Collision muette : hors board → reste en place.
    """
    def __init__(self, N=5):
        self.N = N
        self.agent  = (0, 0)
        self.target = (0, 0)

    def reset(self):
        N = self.N
        self.agent  = (np.random.randint(N), np.random.randint(N))
        self.target = (np.random.randint(N), np.random.randint(N))
        while self.target == self.agent:
            self.target = (np.random.randint(N), np.random.randint(N))
        return self.agent, self.target

    def step(self, action):
        dc, dr = DELTAS[action]
        nc = self.agent[0] + dc
        nr = self.agent[1] + dr
        if 0 <= nc < self.N and 0 <= nr < self.N:
            self.agent = (nc, nr)
        done = (self.agent == self.target)
        return self.agent, self.target, done
```

---

## 4. Code source complet — `src/model.py`

Note pour chatgpt : le modèle ci dessous doit être modifié pour respecter l'architecture du LeWM. 

Voici l'architecture telle qu'expliqué dans le papier : 

'''Model Architecture.
LeWM is built upon two components: an encoder and a predictor. The encoder maps a given frame observation 
𝒐
t
 into a compact, low-dimensional latent representation 
𝒛
t
. The predictor models the environment dynamics in latent space by predicting the embedding of the next frame observation 
𝒛
^
t
+
1
 given the latent embedding 
𝒛
t
 and an action 
𝒂
t
.

Encoder:	
𝒛
t
=
enc
θ
​
(
𝒐
t
)
(LeWM)
Predictor:	
𝒛
^
t
+
1
=
pred
ϕ
​
(
𝒛
t
,
𝒂
t
)
The encoder is implemented as a Vision Transformer (ViT) [16]. Unless otherwise specified, we use the tiny configuration (
∼
5M parameters) with a patch size of 14, 12 layers, 3 attention heads, and hidden dimensions of 192. The observation embedding 
𝒛
t
 is constructed from the [CLS] token embedding of the last layer, followed by a projection step. The projection step maps the [CLS] token embedding into a new representation space using a 1-layer MLP with Batch Normalization [32]. This step is necessary because the final ViT layer applies a Layer Normalization [4], which prevents our anti-collapse objective from being optimized effectively.

The predictor is a transformer with 6 layers, 16 attention heads, and 10% dropout (
∼
10M parameters). Actions are incorporated into the predictor through Adaptive Layer Normalization (AdaLN) [45] applied at each layer. The AdaLN parameters are initialized to zero to stabilize training and ensure that action conditioning impacts the predictor training progressively. The predictor takes as input a history of 
N
 frame representations and predicts the next frame representation auto-regressively with temporal causal masking to avoid looking at future embeddings. The predictor is also followed by a projector network with the same implementation as the one used for the encoder. All components of our world model are learned jointly using the loss described in the following paragraph.'''
 
 adapte les paramètres pour que ça tourne sur un Tesla T4 avec 16Go de RAM si besoin. 
 
 Comme tu vois, l'encoder est à modifier. 

aussi l'action doit etre passé dans Adaptive Layer Normalization (AdaLN)

de plus il y a aussi le sigreg a modifier pour qu'il colle exactement 

fin de la note à chatgpt

```python
"""
model.py — LeWorldModel : Encoder CNN + Predictor MLP + SIGReg
==============================================================
Architecture CNN (Option A — simple, itération rapide) :
  Encoder  : Conv 1→32→64→128→256 + BN + projector MLP + BN1d  (~4.6M params)
  Predictor: MLP (z_t + action_emb) → z_t+1                     (~29k params)
  SIGReg   : Sketched Isotropic Gaussian Regularizer

Fidèle au papier LeWorldModel (arxiv 2603.19312) :
  - PAS de stop-gradient / detach
  - PAS d'EMA / target encoder
  - Gradients propagés partout (end-to-end)
  - Anti-collapse via SIGReg uniquement
  - BatchNorm1d après projector (nécessaire pour SIGReg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    CNN : image (1, 128, 128) → latent (D,)
    4 couches Conv + BN + ReLU → 8×8×256 → projector MLP + BN1d.
    Le BN1d final est crucial pour que SIGReg fonctionne.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1,   32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),  # 64×64
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),  # 32×32
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),  # 16×16
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  # 8×8
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),  # crucial pour SIGReg
        )

    def forward(self, x):
        # x : (B, 1, 128, 128), float32 dans [0, 1]
        return self.projector(self.conv(x))  # (B, latent_dim)


# ── ActionEncoder ─────────────────────────────────────────────────────────────

class ActionEncoder(nn.Module):
    def __init__(self, n_actions=4, action_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_actions, action_dim)

    def forward(self, a):
        # a : (B,) long
        return self.embedding(a)  # (B, action_dim)


# ── Predictor ─────────────────────────────────────────────────────────────────

class Predictor(nn.Module):
    """MLP : (z_t ‖ a_emb) → z_t+1_hat"""
    def __init__(self, latent_dim=32, action_dim=32):
        super().__init__()
        in_dim = latent_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),    nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, z, a_emb):
        return self.net(torch.cat([z, a_emb], dim=-1))


# ── WorldModel ────────────────────────────────────────────────────────────────

class WorldModel(nn.Module):
    def __init__(self, latent_dim=32, action_dim=32, n_actions=4):
        super().__init__()
        self.encoder        = Encoder(latent_dim)
        self.action_encoder = ActionEncoder(n_actions, action_dim)
        self.predictor      = Predictor(latent_dim, action_dim)
        self.latent_dim     = latent_dim

    def forward(self, obs_t, action, obs_t1):
        """
        obs_t, obs_t1 : (B, 1, 128, 128) float32 [0, 1]
        action        : (B,) long

        Retourne z_t, z_t1_hat, z_t1 — gradients actifs partout.
        """
        z_t      = self.encoder(obs_t)
        a_emb    = self.action_encoder(action)
        z_t1_hat = self.predictor(z_t, a_emb)
        z_t1     = self.encoder(obs_t1)   # pas de detach — end-to-end
        return z_t, z_t1_hat, z_t1

    def encode(self, obs):
        return self.encoder(obs)


# ── SIGReg ────────────────────────────────────────────────────────────────────

def sigreg(Z, M=64, n_nodes=50, lam=1.0):
    """
    Sketched Isotropic Gaussian Regularizer.
    Pousse la distribution des latents vers N(0, I).

    Algorithme (appendice A du papier arxiv 2603.19312) :
      1. M directions aléatoires unitaires u ∈ S^{d-1}
      2. Projections h^(m) = Z @ u^(m)  → (B, M)
      3. Test Epps-Pulley : compare ECF empirique à ECF de N(0,1)
      4. Intégrale via quadrature trapèze sur [0.2, 4]

    Z      : (B, D) latents
    M      : nombre de directions (64 local / 1024 papier — impact marginal)
    n_nodes: points de quadrature trapèze
    lam    : largeur du poids gaussien w(t) = exp(-t²/2λ²)
    """
    B, D = Z.shape
    device = Z.device

    # M directions unitaires aléatoires
    U = torch.randn(D, M, device=device)
    U = U / U.norm(dim=0, keepdim=True)          # (D, M)

    # Projections
    H = Z @ U                                     # (B, M)

    # Noeuds de quadrature
    t = torch.linspace(0.2, 4.0, n_nodes, device=device)   # (T,)

    # ECF empirique : phi_N(t) = mean_n(exp(i*t*h_n))
    th       = t.view(1, 1, -1) * H.view(B, M, 1)  # (B, M, T)
    phi_real = th.cos().mean(dim=0)                  # (M, T)
    phi_imag = th.sin().mean(dim=0)                  # (M, T)

    # Cible N(0,1) : phi_0(t) = exp(-t²/2)
    phi0 = (-0.5 * t ** 2).exp()                    # (T,)

    # |phi_N - phi_0|²
    diff2 = (phi_real - phi0.view(1, -1)) ** 2 + phi_imag ** 2  # (M, T)

    # Poids w(t) = exp(-t²/2λ²)
    w = (-0.5 * (t / lam) ** 2).exp()              # (T,)

    # Intégrale trapèze → scalaire par direction → moyenne
    T_m = torch.trapezoid(w * diff2, t, dim=-1)    # (M,)
    return T_m.mean()


# ── Loss totale ───────────────────────────────────────────────────────────────

def total_loss(z_t, z_t1_hat, z_t1, lam=0.1, M=64):
    """
    Loss = MSE(z_t+1_hat, z_t+1) + λ * SIGReg(z_t)

    lam  : poids de SIGReg (λ=0.1 dans le papier — seul vrai hyperparamètre)
    M    : directions SIGReg (64 suffit en local)
    """
    l_pred   = F.mse_loss(z_t1_hat, z_t1)
    l_sigreg = sigreg(z_t, M=M)
    l_total  = l_pred + lam * l_sigreg
    return l_total, l_pred, l_sigreg


# ── Infos ─────────────────────────────────────────────────────────────────────

def model_summary(model):
    n_enc  = sum(p.numel() for p in model.encoder.parameters())
    n_act  = sum(p.numel() for p in model.action_encoder.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    n_tot  = n_enc + n_act + n_pred
    print(f"WorldModel LeWM — {n_tot:,} paramètres")
    print(f"  Encoder        : {n_enc:,}")
    print(f"  ActionEncoder  : {n_act:,}")
    print(f"  Predictor      : {n_pred:,}")
    print(f"  Anti-collapse  : SIGReg (pas d'EMA, pas de detach)")
```

---

## 5. Cellule Setup Colab (à mettre en 1er dans chaque notebook)

```python
# ── Détection environnement ───────────────────────────────────────────────────
import sys, os

IN_COLAB = 'google.colab' in sys.modules
REPO_URL  = "https://github.com/TON_USER/lewm-iso.git"   # ← à mettre à jour
REPO_DIR  = "/content/lewm-iso" if IN_COLAB else "."

if IN_COLAB:
    if not os.path.exists(REPO_DIR):
        os.system(f"git clone {REPO_URL} {REPO_DIR}")
    else:
        os.system(f"cd {REPO_DIR} && git pull")
    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)

    # Google Drive (sauvegarde dataset + checkpoints)
    SAVE_TO_DRIVE = True
    if SAVE_TO_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        DRIVE_DIR = "/content/drive/MyDrive/lewm-iso"
        os.makedirs(DRIVE_DIR, exist_ok=True)
else:
    DRIVE_DIR = "."

print(f"ENV : {'Colab' if IN_COLAB else 'Local'}  |  cwd : {os.getcwd()}")
```

---

## 6. Notebook `lewm_dataset.ipynb` — structure complète

### Config (tout paramétrable ici)

```python
import os
from src.env import RenderConfig

# ════════════════════════════════════════════════════════════════════
# CONFIG — modifier ici uniquement
# ════════════════════════════════════════════════════════════════════

# Board
N         = 5           # taille N×N
N_ACTIONS = 4

# Dataset
N_TRANSITIONS = 80_000  # nombre de transitions
RESET_EVERY   = 200     # reset épisode toutes les N transitions
SEED          = 42      # reproductibilité

# Rendu — tous les paramètres dans RenderConfig
CFG = RenderConfig(
    N           = N,
    az_deg      = 203,
    el_deg      = 32,
    margin_pct  = 1,
    render_size = 128,
    gray_agent  = 141,
    gray_target = 60,
    pad_agent   = 0.32,
    line_w      = 1.2,
    color_board = (210, 235, 228),
    color_line  = (17, 17, 17),
)

# Sauvegarde
DATASET_NAME = f"dataset_iso_N{N}_{N_TRANSITIONS//1000}k.npz"
DATASET_PATH = os.path.join(DRIVE_DIR if (IN_COLAB and SAVE_TO_DRIVE) else ".", DATASET_NAME)

print(f"Board {N}×{N}  |  {N_TRANSITIONS:,} transitions  |  rendu {CFG.render_size}px")
print(f"Sauvegarde → {DATASET_PATH}")
```

### Génération

```python
import numpy as np
from tqdm.auto import tqdm
from src.env import GridWorld, precalculate_render, render_obs

np.random.seed(SEED)

# Précalcul unique des quads/lignes (x5-10 plus rapide)
cache = precalculate_render(CFG)

# Allocations
obs_t_all    = np.zeros((N_TRANSITIONS, CFG.render_size, CFG.render_size), dtype=np.uint8)
actions_all  = np.zeros(N_TRANSITIONS, dtype=np.int64)
obs_t1_all   = np.zeros((N_TRANSITIONS, CFG.render_size, CFG.render_size), dtype=np.uint8)
agent_t_all  = np.zeros((N_TRANSITIONS, 2), dtype=np.int8)
agent_t1_all = np.zeros((N_TRANSITIONS, 2), dtype=np.int8)
target_all   = np.zeros((N_TRANSITIONS, 2), dtype=np.int8)

env = GridWorld(N=N)
agent, target = env.reset()

for i in tqdm(range(N_TRANSITIONS), desc="Génération dataset"):
    if i % RESET_EVERY == 0:
        agent, target = env.reset()

    obs_t  = render_obs(agent, target, cache)
    action = np.random.randint(N_ACTIONS)
    agent_next, target_next, done = env.step(action)
    obs_t1 = render_obs(agent_next, target_next, cache)

    obs_t_all[i]    = obs_t
    actions_all[i]  = action
    obs_t1_all[i]   = obs_t1
    agent_t_all[i]  = agent
    agent_t1_all[i] = agent_next
    target_all[i]   = target

    agent, target = agent_next, target_next
```

### Stats + visualisations

```python
import matplotlib.pyplot as plt

# Masques par type de transition
agent_moved  = ~np.all(agent_t_all == agent_t1_all, axis=1)
on_target_t  = np.all(agent_t_all  == target_all, axis=1)
on_target_t1 = np.all(agent_t1_all == target_all, axis=1)

mask_collision = ~agent_moved
mask_arrive    = agent_moved & on_target_t1 & ~on_target_t
mask_depart    = agent_moved & on_target_t  & ~on_target_t1
mask_normal    = agent_moved & ~on_target_t1 & ~on_target_t

counts = {
    'Collision bord'    : mask_collision.sum(),
    'Arrivée sur cible' : mask_arrive.sum(),
    'Départ de cible'   : mask_depart.sum(),
    'Déplacement normal': mask_normal.sum(),
}

# — Graphique 1 : camembert + distribution actions
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%',
            colors=['#e74c3c','#2ecc71','#f39c12','#3498db'])
axes[0].set_title('Types de transitions')

masks = [mask_collision, mask_arrive, mask_depart, mask_normal]
labels = ['Collision', 'Arrivée', 'Départ', 'Normal']
ACTION_NAMES = {0:'↑', 1:'↓', 2:'←', 3:'→'}
x, width = np.arange(N_ACTIONS), 0.2
for i, (mask, label) in enumerate(zip(masks, labels)):
    if mask.sum() == 0: continue
    dist = [(actions_all[mask] == a).sum() / mask.sum() for a in range(N_ACTIONS)]
    axes[1].bar(x + i*width, dist, width, label=label)
axes[1].set_xticks(x + width*1.5)
axes[1].set_xticklabels([ACTION_NAMES[a] for a in range(N_ACTIONS)])
axes[1].set_title('Distribution actions par type')
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

# — Graphique 2 : heatmaps positions agent et cible
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
hm_agent = np.zeros((N, N))
hm_target = np.zeros((N, N))
for (c, r) in agent_t_all:
    hm_agent[r, c] += 1
for (c, r) in target_all:
    hm_target[r, c] += 1
axes[0].imshow(hm_agent,  cmap='Blues');  axes[0].set_title('Positions agent')
axes[1].imshow(hm_target, cmap='Oranges'); axes[1].set_title('Positions cible')
for ax in axes:
    for r in range(N):
        for c in range(N):
            ax.text(c, r, f'{int(hm_agent[r,c] if ax==axes[0] else hm_target[r,c]):,}',
                    ha='center', va='center', fontsize=7)
plt.suptitle('Distribution spatiale du dataset')
plt.tight_layout(); plt.show()

# — Graphique 3 : exemples visuels par type
type_masks = {'Collision': mask_collision, 'Arrivée': mask_arrive,
              'Départ': mask_depart, 'Normal': mask_normal}
fig, axes = plt.subplots(4, 2, figsize=(6, 14))
for i, (label, mask) in enumerate(type_masks.items()):
    if mask.sum() == 0:
        axes[i][0].axis('off'); axes[i][1].axis('off'); continue
    idx = np.where(mask)[0][0]
    action = actions_all[idx]
    axes[i][0].imshow(obs_t_all[idx],  cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[i][0].set_title(f'{label}\nobs_t  agent={tuple(agent_t_all[idx])}', fontsize=7)
    axes[i][0].axis('off')
    axes[i][1].imshow(obs_t1_all[idx], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[i][1].set_title(f'obs_t+1  action={ACTION_NAMES[action]}\nagent={tuple(agent_t1_all[idx])}', fontsize=7)
    axes[i][1].axis('off')
plt.suptitle('Un exemple de chaque type')
plt.tight_layout(); plt.show()

# Résumé texte
print(f'\n{"Type":25s}  {"Count":>8}  {"Proportion":>10}')
print('-' * 48)
for label, count in counts.items():
    print(f'{label:25s}  {count:>8}  {count/N_TRANSITIONS*100:>9.1f}%')
```

### Sauvegarde

```python
np.savez_compressed(
    DATASET_PATH,
    obs_t    = obs_t_all,
    actions  = actions_all,
    obs_t1   = obs_t1_all,
    agent_t  = agent_t_all,
    agent_t1 = agent_t1_all,
    target   = target_all,
)
size_mb = os.path.getsize(DATASET_PATH) / 1e6
print(f"✓ {DATASET_PATH}  ({size_mb:.1f} Mo)")
print(f"  obs_t    : {obs_t_all.shape}  {obs_t_all.dtype}")
print(f"  actions  : {actions_all.shape}  {actions_all.dtype}")
print(f"  agent_t  : {agent_t_all.shape}  {agent_t_all.dtype}")
```

---

## 7. Notebook `lewm_train.ipynb` — structure complète

### Config entraînement

```python
import os, torch
from src.env import RenderConfig  # pour retrouver les params si besoin

# ════════════════════════════════════════════════════════════════════
# CONFIG ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════════════

# Dataset
N               = 5
DATASET_NAME    = f"dataset_iso_N{N}_80k.npz"
DATASET_PATH    = os.path.join(DRIVE_DIR if (IN_COLAB and SAVE_TO_DRIVE) else ".", DATASET_NAME)

# Modèle
LATENT_DIM  = 32
ACTION_DIM  = 32
N_ACTIONS   = 4

# Training
BATCH_SIZE  = 256
N_EPOCHS    = 50
LR          = 1e-4
VAL_SPLIT   = 0.1     # 10% validation

# Loss
LAMBDA_REG  = 0.1     # poids SIGReg (λ dans le papier)
M_SIGREG    = 64      # directions (64 local / 1024 papier)

# Scheduler
LR_STEP     = 10      # StepLR step_size
LR_GAMMA    = 0.5     # StepLR gamma

# Checkpoints
CKPT_DIR    = os.path.join(DRIVE_DIR if (IN_COLAB and SAVE_TO_DRIVE) else ".", "checkpoints")
CKPT_EVERY  = 5       # sauvegarder tous les N epochs
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"GPU : {torch.cuda.get_device_name(0)}")
else:
    print("CPU — entraînement lent, préférer Colab GPU")
```

### Dataset PyTorch

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class TransitionDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        # Normalisation uint8 [0,255] → float32 [0,1]
        self.obs_t   = data['obs_t'].astype(np.float32)  / 255.0
        self.actions = data['actions'].astype(np.int64)
        self.obs_t1  = data['obs_t1'].astype(np.float32) / 255.0

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.obs_t[idx]).unsqueeze(0),   # (1, H, W)
            torch.tensor(self.actions[idx]),               # scalaire long
            torch.tensor(self.obs_t1[idx]).unsqueeze(0),  # (1, H, W)
        )

dataset = TransitionDataset(DATASET_PATH)
n_val   = int(len(dataset) * VAL_SPLIT)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=(DEVICE=="cuda"))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(DEVICE=="cuda"))
print(f"Train : {n_train:,}  |  Val : {n_val:,}  |  Batchs/epoch : {len(train_loader)}")
```

### Instanciation + boucle

```python
from src.model import WorldModel, total_loss, model_summary

model     = WorldModel(LATENT_DIM, ACTION_DIM, N_ACTIONS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)
model_summary(model)

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot, tot_pred, tot_sig = 0., 0., 0.
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for obs_t, action, obs_t1 in loader:
            obs_t  = obs_t.to(DEVICE)
            action = action.to(DEVICE)
            obs_t1 = obs_t1.to(DEVICE)
            z_t, z_t1_hat, z_t1 = model(obs_t, action, obs_t1)
            l, l_pred, l_sig = total_loss(z_t, z_t1_hat, z_t1, lam=LAMBDA_REG, M=M_SIGREG)
            if train:
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            tot      += l.item()
            tot_pred += l_pred.item()
            tot_sig  += l_sig.item()
    n = len(loader)
    return tot/n, tot_pred/n, tot_sig/n

import time
history = {'train':[], 'val':[], 'mse':[], 'sigreg':[]}
best_val = float('inf')

print(f"{'Ep':>4}  {'Train':>8}  {'Val':>8}  {'MSE':>8}  {'SIGReg':>8}  {'lr':>8}  {'Time':>6}")
print('─' * 70)

for ep in range(1, N_EPOCHS + 1):
    t0 = time.time()
    tr, tr_mse, tr_sig = run_epoch(train_loader, train=True)
    val, _, _          = run_epoch(val_loader,   train=False)
    scheduler.step()
    elapsed = time.time() - t0
    lr_now = scheduler.get_last_lr()[0]

    print(f"{ep:>4}  {tr:>8.4f}  {val:>8.4f}  {tr_mse:>8.4f}  {tr_sig:>8.4f}  {lr_now:>8.2e}  {elapsed:>5.0f}s")

    history['train'].append(tr)
    history['val'].append(val)
    history['mse'].append(tr_mse)
    history['sigreg'].append(tr_sig)

    if ep % CKPT_EVERY == 0:
        torch.save({'epoch': ep, 'model': model.state_dict(),
                    'config': {'latent_dim': LATENT_DIM, 'action_dim': ACTION_DIM,
                               'n_actions': N_ACTIONS, 'lambda': LAMBDA_REG}},
                   os.path.join(CKPT_DIR, f'model_ep{ep:03d}.pt'))
    if val < best_val:
        best_val = val
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'model_best.pt'))

print(f"\n✓ Terminé. Meilleur val : {best_val:.4f}")
```

### Courbes de loss + vérification collapse

```python
# — Loss curves
ep = range(1, len(history['train']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep, history['train'], label='train')
axes[0].plot(ep, history['val'],   label='val', linestyle='--')
axes[0].set_title('Loss totale'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep, history['mse'],    label='MSE pred')
axes[1].plot(ep, history['sigreg'], label='SIGReg')
axes[1].set_title('Composantes (train)'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# — Vérification anti-collapse
model.eval()
obs_t, action, obs_t1 = next(iter(val_loader))
with torch.no_grad():
    z = model.encode(obs_t.to(DEVICE)).cpu()
var_per_dim = z.var(dim=0).numpy()
fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(range(LATENT_DIM), var_per_dim)
ax.axhline(1.0, color='red', linestyle='--', label='cible N(0,1)')
ax.set_xlabel('Dimension latente'); ax.set_ylabel('Variance')
ax.set_title('Variance par dim — collapse si tout ≈ 0')
ax.legend(); plt.tight_layout(); plt.show()
print(f"Variance moyenne : {var_per_dim.mean():.4f}")
print(f"Dimensions mortes (var < 0.01) : {(var_per_dim < 0.01).sum()}/{LATENT_DIM}")
```

---

## 8. `requirements.txt`

```
numpy
matplotlib
pillow
torch
tqdm
```

Colab a déjà torch+cuda, PIL, numpy, matplotlib, tqdm. Rien à installer normalement.

---

## 9. `.gitignore`

```
*.npz
checkpoints/
logs/
__pycache__/
.ipynb_checkpoints/
*.pt
*.pyc
```

note à chatgpt

ce worklflow est à préciser. En effet, je ne suis pas familier avec l'utilisation de google colab d'une part ni de l'utilisation de google colab avec git clone. 
donc parlons en 

fin de la note

---

## 10. Workflow local ↔ Colab

```
LOCAL (JupyterLab)                      COLAB
──────────────────────────────────────────────────────────────
1. éditer src/env.py ou src/model.py
2. tester en local (petit N, peu de transitions)
3. git commit + push
                                   4. git clone (1ère fois)
                                      ou git pull (cellule 0)
                                   5. générer dataset → Drive
                                   6. entraîner → checkpoints Drive
                                   7. télécharger checkpoints
8. analyser / visualiser local
9. itérer
```

**Règle** : on ne modifie jamais les fichiers directement dans Colab.
Tout changement = commit local + pull dans Colab.

---

## 11. Ordre de développement

- [x] Planification (ce document)
- [ ] Créer repo `lewm-iso` vide, `.gitignore`, `requirements.txt`
- [ ] Écrire `src/env.py` depuis le code de la section 3
- [ ] Tester `env.py` en local : vérifier rendu visuel + performance
- [ ] Écrire `lewm_dataset.ipynb` depuis la section 6
- [ ] Générer dataset local (petit : N=5, 10k transitions pour test)
- [ ] Écrire `src/model.py` depuis le code de la section 4
- [ ] Écrire `lewm_train.ipynb` depuis la section 7
- [ ] Tester boucle d'entraînement local (1-2 epochs pour valider)
- [ ] Tester Colab end-to-end : clone → dataset → train → checkpoints Drive
- [ ] Visualisation espace latent post-training (PCA 2D, probing positions)

---

## 12. Résultats de référence (session précédente)

Pour valider que l'implémentation fonctionne :
- **Dataset 80k** : ~15-20s de génération avec précalcul, ~X Mo compressé
- **Modèle CNN** : ~4.6M params, ~634ms/batch sur CPU, ~3min/epoch (CPU)
- **Entraînement 20 epochs** : meilleur val ≈ 0.0104 (loss totale)
- **Anti-collapse** : variance moyenne des latents doit être >> 0 après entraînement

Si la variance par dimension est proche de 0 sur toutes les dims → collapse, augmenter λ (LAMBDA_REG).
Si val loss remonte après train loss → overfitting, réduire LR ou augmenter le dataset.

---

*Tout le code dans ce document est copié-collé depuis la session locale validée.
Aucun code n'est inventé — ce qui est ici a été testé.*

Mon Drive/projetColab/wm_colab/test_run

"""
model.py — LeWorldModel (LeWM)
==============================
Encoder  : ViT (patch embedding + Transformer, CLS token)
Predictor: Transformer autorégressif conditionné par actions via AdaLN
Loss     : MSE prédiction + λ·SIGReg anti-collapse

Architecture :
  - patch_size=16  →  8×8 = 64 patches par image 128×128
  - embed_dim=192, n_heads=3 (head_dim=64), mlp_ratio=4
  - encoder_depth=4, predictor_depth=2
  - SIGReg : M=1024 projections aléatoires, λ=0.1
  - Pas de stop-gradient, pas d'EMA

Usage :
    model = LeWorldModel()
    loss, info = model(obs, actions)
    # obs     : (B, T, H, W) float32 in [0, 1]
    # actions : (B,) int64  — action entre obs[:,0] et obs[:,1]
    # info    : dict avec 'total', 'pred', 'sigreg'
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SIGReg ────────────────────────────────────────────────────────────────────

class SIGReg(nn.Module):
    """
    Régularisation anti-collapse par projections aléatoires.

    Projette Z sur M directions aléatoires unitaires et pénalise
    tout écart à une distribution N(0, 1) univariée (moyenne et variance).

    La matrice de projection est un buffer fixe (non entraînable) qui
    se déplace automatiquement avec .to(device).
    """

    def __init__(self, embed_dim: int, M: int = 1024):
        super().__init__()
        proj = torch.randn(embed_dim, M)
        proj = F.normalize(proj, dim=0)          # colonnes unitaires
        self.register_buffer("proj", proj)        # (D, M)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z : (B, D)
        Retourne un scalaire.
        """
        h = Z @ self.proj                         # (B, M)
        mean_penalty = h.mean(dim=0).pow(2).mean()           # E[h] → 0
        var_penalty  = (h.pow(2).mean(dim=0) - 1).pow(2).mean()  # E[h²] → 1
        return mean_penalty + var_penalty


# ── Patch Embedding ───────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Découpe l'image en patches et projette chacun en un vecteur."""

    def __init__(self, img_size: int = 128, patch_size: int = 16, embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size doit être divisible par patch_size"
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 1, H, W) → (B, n_patches, D)
        return self.proj(x).flatten(2).transpose(1, 2)


# ── Bloc Transformer standard (encodeur) ─────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ── Encodeur ViT ─────────────────────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """
    ViT-Tiny : patch embedding + CLS token + Transformer + LayerNorm finale.
    Chaque image 128×128 → vecteur latent de dimension embed_dim.
    """

    def __init__(
        self,
        img_size    : int   = 128,
        patch_size  : int   = 16,
        embed_dim   : int   = 192,
        depth       : int   = 4,
        n_heads     : int   = 3,
        mlp_ratio   : float = 4.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, n_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 1, H, W) → (B, D)
        B = x.shape[0]
        x = self.patch_embed(x)                                    # (B, n_patches, D)
        cls = self.cls_token.expand(B, -1, -1)                     # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                             # (B, 1+n_patches, D)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                              # CLS token → (B, D)


# ── Bloc AdaLN (prédicteur) ───────────────────────────────────────────────────

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN : x * (1 + scale) + shift. Initialisation near-identity à scale≈0."""
    return x * (1 + scale) + shift


class AdaLNBlock(nn.Module):
    """
    Bloc Transformer conditionné par AdaLN.

    Le vecteur d'action (B, D) produit scale et shift pour les deux
    LayerNorm du bloc. Les LN n'ont pas de paramètres appris propres
    (elementwise_affine=False) : c'est entièrement l'AdaLN qui joue ce rôle.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        # action_emb → (scale1, shift1, scale2, shift2)
        self.ada_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim))
        # Init near-zero : les modulations démarrent proches de l'identité
        nn.init.zeros_(self.ada_mod[-1].weight)
        nn.init.zeros_(self.ada_mod[-1].bias)

    def forward(
        self,
        x          : torch.Tensor,   # (B, T, D)
        action_emb : torch.Tensor,   # (B, D)
        mask       : torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Modulation : (B, 4*D) → 4 × (B, D) → unsqueeze pour broadcast sur T
        s1, sh1, s2, sh2 = self.ada_mod(action_emb).chunk(4, dim=-1)
        s1  = s1.unsqueeze(1);  sh1 = sh1.unsqueeze(1)   # (B, 1, D)
        s2  = s2.unsqueeze(1);  sh2 = sh2.unsqueeze(1)

        h = _modulate(self.norm1(x), sh1, s1)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        x = x + self.mlp(_modulate(self.norm2(x), sh2, s2))
        return x


# ── Prédicteur Transformer ────────────────────────────────────────────────────

class TransformerPredictor(nn.Module):
    """
    Prédicteur autorégressif conditionné par action via AdaLN.

    Reçoit la séquence de latents z_{0:T-1} et l'action unique (B,),
    produit les prédictions ẑ_{0:T-1}.

    La loss compare ẑ_{0:T-2} avec z_{1:T-1} (décalage d'un pas).
    """

    def __init__(
        self,
        embed_dim : int = 192,
        depth     : int = 2,
        n_heads   : int = 3,
        n_actions : int = 4,
        mlp_ratio : float = 4.0,
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim doit être pair (PE sinusoïdale)"
        self.embed_dim     = embed_dim
        self.action_embed  = nn.Embedding(n_actions, embed_dim)
        self.blocks = nn.ModuleList([
            AdaLNBlock(embed_dim, n_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

    @staticmethod
    def _sinusoidal_pe(T: int, D: int, device, dtype) -> torch.Tensor:
        pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)       # (T, 1)
        div = torch.exp(
            torch.arange(0, D, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / D)
        )                                                                       # (D/2,)
        pe = torch.zeros(T, D, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)                                                 # (1, T, D)

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        z       : (B, T, D)
        actions : (B,) int64  — une seule action pour T=2
        →         (B, T, D)
        """
        B, T, D = z.shape
        x = z + self._sinusoidal_pe(T, D, z.device, z.dtype)

        # Expand l'action sur toute la séquence
        action_emb = self.action_embed(actions)                    # (B, D)

        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=z.device), diagonal=1
        )                                                          # masque causal

        for blk in self.blocks:
            x = blk(x, action_emb, mask=mask)

        return self.norm(x)                                        # (B, T, D)


# ── LeWorldModel ─────────────────────────────────────────────────────────────

class LeWorldModel(nn.Module):
    """
    World Model latent appris offline, sans récompense (LeWM).

    Encoder  : ViT — o_t → z_t
    Predictor: Transformer AdaLN — (z_t, a_t) → ẑ_{t+1}
    Loss     : MSE(ẑ_{t+1}, z_{t+1}) + λ·SIGReg(z)

    Entraîné end-to-end, sans stop-gradient ni EMA.
    """

    def __init__(
        self,
        img_size        : int   = 128,
        patch_size      : int   = 16,
        embed_dim       : int   = 192,
        encoder_depth   : int   = 4,
        predictor_depth : int   = 2,
        n_heads         : int   = 3,
        n_actions       : int   = 4,
        mlp_ratio       : float = 4.0,
        sigreg_M        : int   = 1024,
        sigreg_lambda   : float = 0.1,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=encoder_depth,
            n_heads=n_heads, mlp_ratio=mlp_ratio,
        )
        self.predictor = TransformerPredictor(
            embed_dim=embed_dim, depth=predictor_depth,
            n_heads=n_heads, n_actions=n_actions, mlp_ratio=mlp_ratio,
        )
        self.sigreg        = SIGReg(embed_dim, M=sigreg_M)
        self.sigreg_lambda = sigreg_lambda

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs : (B, T, H, W) float32 in [0, 1]
        →     (B, T, D)
        """
        B, T, H, W = obs.shape
        x = obs.reshape(B * T, 1, H, W)
        emb = self.encoder(x)             # (B*T, D)
        return emb.view(B, T, -1)

    def forward(
        self,
        obs     : torch.Tensor,   # (B, T, H, W) float32 in [0, 1]
        actions : torch.Tensor,   # (B,) int64
    ) -> tuple[torch.Tensor, dict]:
        """
        Retourne (loss_totale, info_dict).
        info_dict : {'total': ..., 'pred': ..., 'sigreg': ...}
        """
        emb  = self.encode(obs)               # (B, T, D)
        pred = self.predictor(emb, actions)   # (B, T, D)

        # Loss prédiction : ẑ_t vs z_{t+1}, sans stop-gradient
        pred_loss = F.mse_loss(pred[:, :-1], emb[:, 1:])

        # SIGReg : sur chaque pas de temps, moyenné
        T = emb.shape[1]
        reg_losses = [self.sigreg(emb[:, t]) for t in range(T)]
        reg_loss   = torch.stack(reg_losses).mean()

        total = pred_loss + self.sigreg_lambda * reg_loss

        info = {
            "total"  : total.item(),
            "pred"   : pred_loss.item(),
            "sigreg" : reg_loss.item(),
        }
        return total, info

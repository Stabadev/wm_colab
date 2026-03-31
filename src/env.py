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

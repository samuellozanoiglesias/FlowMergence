"""
sim_gif.py -- illustrative re-implementation of the FlowMergence movement rules, for animations.

What it implements (as in the paper's model):
  * periodic L x L lattice, two layers (BML): horizontal agents move right on even steps,
    vertical agents move up on odd steps;
  * hard exclusion within a lane: a move needs the target cell empty in the agent's own layer
    at the start of the step;
  * crossing conflict: if the target cell is occupied in the OTHER layer, the move is blocked
    with probability lambda (lambda*(1-q)^k inside the zone of k coordinators);
  * jobs of fixed length; with probability c a finished job must be carried to the hub line
    (column x = L/2 for horizontal agents, row y = L/2 for vertical ones) before it counts;
  * fixed coordinators ("police") that sit still, cover a (2r+1)^2 zone and relocate to
    recently congested cells with probability 0.1 per step.
What it omits (for clarity): response thresholds, learning and spillovers (jobs have a fixed
length here), endogenous recruitment of coordinators. It is NOT the code that produced the
paper's results; it only visualizes the rules.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG = (246, 244, 239); GRID = (226, 223, 214); INK = (27, 31, 36); MUTED = (95, 101, 110)
HUB = (226, 216, 242); HUBE = (106, 81, 163)
H_COL = [(43, 108, 176), (125, 170, 220)]     # horizontal tasks 0, 2
V_COL = [(217, 130, 43), (240, 186, 120)]     # vertical tasks 1, 3
RED = (200, 40, 40); COORD = (46, 139, 87); COORDZ = (214, 236, 222)


class Sim:
    def __init__(self, L, rho, c, lam, job=12, n_coord=0, r=3, q=0.5, seed=0):
        self.L, self.c, self.lam, self.job, self.r, self.q = L, c, lam, job, r, q
        self.rng = np.random.default_rng(seed)
        N = int(rho * L * L)
        cells = self.rng.choice(L * L, size=N, replace=False)   # one agent per cell at start
        self.x = (cells % L).astype(int); self.y = (cells // L).astype(int)
        self.layer = np.zeros(N, int); self.layer[N // 2:] = 1   # 0 = horizontal, 1 = vertical
        self.rng.shuffle(self.layer)
        self.task = np.where(self.layer == 0, self.rng.choice([0, 2], N), self.rng.choice([1, 3], N))
        self.left = self.rng.integers(1, job + 1, N)              # moves left in the current job
        self.deliv = np.zeros(N, bool)
        self.blocked = np.zeros(N, bool)
        self.done = 0; self.t = 0; self.t_deliv = 0; self.t_total = 0
        self.cx = self.rng.integers(0, L, n_coord); self.cy = self.rng.integers(0, L, n_coord)
        self.heat = np.zeros((L, L))
        self.N = N; self.hub = L // 2

    def coord_k(self):
        L, r = self.L, self.r
        k = np.zeros((L, L), int)
        for cx, cy in zip(self.cx, self.cy):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    k[(cx + dx) % L, (cy + dy) % L] += 1
        return k

    def step(self):
        L = self.L
        mover = self.t % 2                                        # layer that moves this step
        occ = np.zeros((2, L, L), bool)
        occ[self.layer, self.x, self.y] = True
        k = self.coord_k()
        idx = np.where(self.layer == mover)[0]
        tx = (self.x[idx] + (mover == 0)) % L
        ty = (self.y[idx] + (mover == 1)) % L
        own = occ[mover, tx, ty]
        cross = occ[1 - mover, tx, ty]
        p_block = self.lam * (1 - self.q) ** k[tx, ty]
        cross_block = cross & (self.rng.random(len(idx)) < p_block)
        ok = ~own & ~cross_block
        self.blocked[:] = False
        self.blocked[idx[own | cross_block]] = True
        self.heat *= 0.98
        np.add.at(self.heat, (self.x[idx[~ok]], self.y[idx[~ok]]), 1.0)
        mv = idx[ok]
        self.x[mv] = tx[ok]; self.y[mv] = ty[ok]
        # progress of movers
        work = mv[~self.deliv[mv]]
        self.left[work] -= 1
        fin = work[self.left[work] <= 0]
        to_hub = fin[self.rng.random(len(fin)) < self.c]
        self.deliv[to_hub] = True
        local = np.setdiff1d(fin, to_hub)
        self.done += len(local); self.left[local] = self.job
        d = mv[self.deliv[mv]]
        arrived = d[np.where(self.layer[d] == 0, self.x[d] == self.hub, self.y[d] == self.hub)]
        self.done += len(arrived); self.deliv[arrived] = False; self.left[arrived] = self.job
        # Gamma bookkeeping (agents of the moving layer)
        self.t_deliv += self.deliv[idx].sum(); self.t_total += len(idx)
        # coordinators relocate to congested cells
        for i in range(len(self.cx)):
            if self.rng.random() < 0.1:
                flat = self.heat.ravel()
                j = self.rng.choice(L * L, p=(flat + 1e-9) / (flat + 1e-9).sum())
                self.cx[i], self.cy[i] = j % L, j // L
        self.t += 1

    def gamma(self):
        d = self.t_deliv; o = self.t_total - d
        return d / o if o > 0 else 0.0


def tri(draw, cx, cy, s, layer, fill, outline, w=1):
    if layer == 0:   # right-pointing
        pts = [(cx - s, cy - s), (cx + s, cy), (cx - s, cy + s)]
    else:            # up-pointing (image y grows downward)
        pts = [(cx - s, cy + s), (cx, cy - s), (cx + s, cy + s)]
    draw.polygon(pts, fill=fill, outline=outline, width=w)


def render(sim, cell, ox, oy, draw, show_heat=False):
    L = sim.L; Wp = L * cell
    draw.rectangle([ox, oy, ox + Wp, oy + Wp], fill=BG)
    if len(sim.cx):
        k = sim.coord_k()
        for i in range(L):
            for j in range(L):
                if k[i, j]:
                    X = ox + i * cell; Y = oy + (L - 1 - j) * cell
                    draw.rectangle([X, Y, X + cell, Y + cell], fill=COORDZ)
    h = sim.hub
    draw.rectangle([ox + h * cell, oy, ox + (h + 1) * cell, oy + Wp], fill=HUB)
    draw.rectangle([ox, oy + (L - 1 - h) * cell, ox + Wp, oy + (L - h) * cell], fill=HUB)
    for i in range(L + 1):
        draw.line([ox + i * cell, oy, ox + i * cell, oy + Wp], fill=GRID)
        draw.line([ox, oy + i * cell, ox + Wp, oy + i * cell], fill=GRID)
    s = cell * 0.36
    for a in range(sim.N):
        X = ox + (sim.x[a] + 0.5) * cell + (-cell * 0.12 if sim.layer[a] == 1 else cell * 0.06)
        Y = oy + (L - 1 - sim.y[a] + 0.5) * cell
        col = H_COL[sim.task[a] // 2] if sim.layer[a] == 0 else V_COL[(sim.task[a] - 1) // 2]
        if sim.deliv[a]:
            tri(draw, X, Y, s, sim.layer[a], BG, col, w=max(1, cell // 7))
        else:
            tri(draw, X, Y, s, sim.layer[a], col, RED if sim.blocked[a] else INK,
                w=max(1, cell // 7) if sim.blocked[a] else 1)
    for cx, cy in zip(sim.cx, sim.cy):
        X = ox + (cx + 0.5) * cell; Y = oy + (L - 1 - cy + 0.5) * cell
        rr = cell * 0.55
        pts = []
        for m in range(10):
            ang = np.pi / 2 + m * np.pi / 5
            rad = rr if m % 2 == 0 else rr * 0.45
            pts.append((X + rad * np.cos(ang), Y - rad * np.sin(ang)))
        draw.polygon(pts, fill=COORD, outline=INK)
    draw.rectangle([ox, oy, ox + Wp, oy + Wp], outline=MUTED, width=2)


def make_pair(fn, sims, titles, steps=320, every=2, cell=10, warm=60, dur=90, foot=None):
    for s in sims:
        for _ in range(warm):
            s.step()
        s.done = 0; s.t_deliv = 0; s.t_total = 0; s.t0 = s.t
    L = sims[0].L; pad = 30; top = 104; bottom = 92 if foot else 60
    W = len(sims) * L * cell + (len(sims) + 1) * pad; H = top + L * cell + bottom
    fT = ImageFont.truetype(FONTB, 22); fS = ImageFont.truetype(FONT, 18); fF = ImageFont.truetype(FONT, 15)
    frames = []
    for f in range(steps // every):
        for _ in range(every):
            for s in sims:
                s.step()
        im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
        for i, (s, t) in enumerate(zip(sims, titles)):
            ox = pad + i * (L * cell + pad)
            d.text((ox, 18), t, font=fT, fill=INK)
            el = (s.t - s.t0) / 2
            rate = s.done / s.N / max(el, 1) * 100
            d.text((ox, 50), f"output: {rate:4.1f} jobs / agent / 100 sweeps", font=fS, fill=MUTED)
            d.text((ox, 72), f"Γ (time delivering / other time) = {s.gamma():.2f}", font=fS, fill=MUTED)
            render(s, cell, ox, top, d)
        if foot:
            d.text((pad, top + L * cell + 14), foot[0], font=fF, fill=MUTED)
            d.text((pad, top + L * cell + 36), foot[1], font=fF, fill=MUTED)
            d.text((pad, top + L * cell + 58), foot[2], font=fF, fill=MUTED)
        frames.append(im.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    frames[0].save(fn, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
    return [(s.done / s.N / ((s.t - s.t0) / 2) * 100, s.gamma()) for s in sims]


if __name__ == "__main__":
    out = "/mnt/user-data/outputs/deck_assets/"
    # 1) decentralized vs centralized, same initial state (rho = 0.22, lambda = 0.5)
    print("c0 vs c1:", make_pair(out + "gif_c0_vs_c1.gif",
          [Sim(40, 0.22, 0.0, 0.5, seed=3), Sim(40, 0.22, 1.0, 0.5, seed=3)],
          ["c = 0: deliver locally", "c = 1: every job to the hub"]))
    # 2) coordinators in a congested, centralized system (rho = 0.42, lambda = 1)
    print("coord:", make_pair(out + "gif_coordinators.gif",
          [Sim(40, 0.42, 1.0, 1.0, seed=2, n_coord=0), Sim(40, 0.42, 1.0, 1.0, seed=2, n_coord=24)],
          ["No coordinators", "24 coordinators (~4%)"], steps=480, every=4, warm=100))
    # 3) close-up of the rules
    print("closeup:", make_pair(out + "gif_rules_closeup.gif",
          [Sim(14, 0.28, 0.5, 0.5, job=6, seed=11, n_coord=1, r=2)],
          ["Close-up, c = 0.5"], steps=160, every=1, cell=34, warm=10, dur=260))

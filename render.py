"""Render the logo-animation frames from the LATA dumps and build the GIF.

Forward simulation: letters -> dispersed bubbles. The GIF is assembled in
reverse (bubbles -> logo), with a hold on the final logo frame.

Usage: python3 render.py [case_name] [t_max] (default: logo, all dumps)
"""
import math
import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import lata
from make_letters import glyph_colors

# the periodic box; overwritten in main() from the case's own LATA domain
DOM = (2.56, 0.64, 1.28)

LIGHT = np.array([-0.3, -0.8, 0.5])
BG = '#0d1b2a'          # deep blue background, water-ish

# -------------------------------------------------------------- output knobs
RENDER_DPI = 150        # 150 -> 1920x960 frame PNGs
GIF_SIZE = (960, 480)   # the GIF is downscaled; the MP4 keeps full size
MP4_CRF = 17            # x264 quality (lower = better)

# ------------------------------------------------------------ camera knobs
CAM_ELEV = 12           # degrees above the horizon
CAM_AZIM = -76          # -90 = face-on; toward -75 brings the nabla side
#                         closer and pushes the T away (yaw about z)
CAM_FOCAL = 0.5         # matplotlib focal_length: smaller = more perspective
CAM_AZIM_SWEEP = 6      # degrees of slow yaw over the movie: the reversed
#                         playback starts at CAM_AZIM + sweep (bubble cloud)
#                         and settles on CAM_AZIM as the logo locks in. 0 = off
START_DUMP = 1          # 0 = include the initial condition (crisp CAD look);
#                         1 = start at the first simulated dump
FRAME_MS = 70           # GIF frame duration

# ------------------------------------------------------ camera-follow knobs
# The camera tracks the bubble cloud: each frame is centred on the (smoothed)
# cloud bounding box and zoomed to it, so the frame never holds dead space.
# In the reversed playback the camera glides up with the rising swarm and
# pulls back as the logo assembles. CAM_FOLLOW = 0 restores the fixed frame.
CAM_FOLLOW = 1
CAM_SMOOTH = 15         # smoothing window (frames) of the camera path
CAM_MARGIN = 0.22       # relative margin around the cloud bounding box

# -------------------------------------------------------- background knobs
# The background is the frame of the water the swarm is carried in. It
# drifts at BG_DRIFT (m/s, upward in forward time): in the reversed playback
# it streams downward past the swarm, so the whole shot reads as ascending —
# the common carriage the buoyant mode's still liquid (and the
# centroid-pinned camera) otherwise hide. Two thematic layers:
#  - the IJK grid: a faint structured-mesh plane behind everything (this is
#    a mesh code; the bubbles rise through their own grid). Horizontal
#    k-lines dominate — discrete floors sweeping past.
#  - micro-bubbles: parallax layers of small out-of-focus bubbles rising
#    with the flow, each layer at its own apparent speed (larger background
#    bubbles rise faster relative to the water, so they stream down slower
#    in playback), with a slight lateral wobble.
BG_SEED = 7
BG_RGB = (0.62, 0.72, 0.85)
# The grid is the reference frame of the shot; everything ascends against
# it. ASCEND is the playback ascent speed (m/s) of the logo swarm + camera:
# a uniform lift added to the rendered positions, i.e. the mean upward
# carriage of the liquid the simulation deliberately does not carry. The
# micro-bubble layers ascend too, slower than the swarm (they sink relative
# to the tracking camera, but rise against the grid).
ASCEND = 0.4
GRID_STRIDE = 8         # mesh cells between grid lines (0 disables the grid)
GRID_DX = 0.02          # the mesh spacing the grid depicts
GRID_ALPHA = 0.22
GRID_Y = 1.35           # depth of the grid plane (behind everything)
GRID_LABELS = True      # a few faint k-indices on the horizontal lines
# per layer: (count, y0, y1, size_min, size_max, alpha, playback ascend
# speed in m/s, wobble amplitude in m). Nearer layers are larger and faster
# (bigger bubbles rise faster); all are slower than the swarm's ~ASCEND.
# Layers with y < 0 draw over the surfaces (in front of the bubble slab).
BUBBLE_LAYERS = (
    (260, 1.00, 1.30, 2.0, 8.0, 0.55, 0.12, 0.015),
    (170, 0.55, 0.95, 4.0, 14.0, 0.65, 0.22, 0.025),
    (45, -0.50, -0.05, 8.0, 26.0, 0.75, 0.32, 0.035),
)

# ------------------------------------------------------------- trail knobs
# Each bubble drags a fading streak along its own trajectory (the tracked
# centroid path): in the reversed playback it hangs below the rising bubble
# and vanishes as the logo locks in. TRAIL_T = 0 disables.
TRAIL_T = 0.0           # seconds of trajectory shown behind each bubble
TRAIL_ALPHA = 0.35      # opacity where the streak leaves the bubble
TRAIL_LW = 2.2          # line width
TRAIL_FADE_T = 0.5      # trails and streamlines melt away over the last
#                         seconds of assembly, so the locked-in logo is clean

# -------------------------------------------------------- streamline knobs
# A curtain of instantaneous streamlines of the simulated liquid velocity,
# seeded on a fixed comb along the bottom of the frame and integrated
# upward until each line comes within STREAM_CLEAR of a bubble surface: the
# flow fills the bubble-free space without ever touching the bubbles. Needs
# `VELOCITY elem` among the deck's post-processed fields. STREAM_COLS = 0
# disables.
STREAM_COLS = 0         # seed columns across the frame bottom
STREAM_LEN = 1.5        # maximum arc length of one line, in m
STREAM_STEP = 0.012     # integration step, in m
STREAM_CLEAR = 0.10     # lines stop this far from the nearest bubble surface
STREAM_ALPHA = 0.28
STREAM_LW = 1.3
STREAM_RGB = (0.55, 0.70, 0.85)   # faint water-blue, distinct from bubbles


def unwrap_component(x, centre, period):
    """Wrap coordinates to within half a period of `centre`."""
    return (x - centre + period / 2) % period + centre - period / 2


def circular_centre(x, period):
    ang = x * (2 * np.pi / period)
    return np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) \
        * period / (2 * np.pi)


class VelocityField:
    """Periodic trilinear sampler of one dump's cell-centred VELOCITY."""

    def __init__(self, master, index):
        axes = master.node_coordinates()
        self.origin = np.array([a[0] for a in axes])
        self.n = np.array([len(a) - 1 for a in axes])        # (ni, nj, nk)
        self.d = np.array([(a[-1] - a[0]) / (len(a) - 1) for a in axes])
        # `VELOCITY elem` is dumped as three ELEM scalars; C ordering of the
        # dump: k outer, then j, then i (verified against the logo geometry)
        comps = [master.field(f'CELL_VELOCITY_{a}', index) for a in 'XYZ']
        self.v = np.stack([c.reshape(self.n[2], self.n[1], self.n[0])
                           for c in comps], axis=-1)
        # subtract the volume-mean drift (the periodic box free-falls as a
        # whole): the streamlines show the flow in the falling frame, where
        # the wakes and the swirl live, not the uniform fall
        self.v -= self.v.mean(axis=(0, 1, 2))

    def sample(self, pts):
        """Velocity at pts [M,3] (any unwrapped coordinates)."""
        u = (pts - self.origin) / self.d - 0.5     # cell-centre units
        i0 = np.floor(u).astype(int)
        f = u - i0
        out = np.zeros_like(pts)
        for di in (0, 1):
            wx = f[:, 0] if di else 1 - f[:, 0]
            ii = (i0[:, 0] + di) % self.n[0]
            for dj in (0, 1):
                wy = f[:, 1] if dj else 1 - f[:, 1]
                jj = (i0[:, 1] + dj) % self.n[1]
                for dk in (0, 1):
                    wz = f[:, 2] if dk else 1 - f[:, 2]
                    kk = (i0[:, 2] + dk) % self.n[2]
                    out += (wx * wy * wz)[:, None] * self.v[kk, jj, ii, :]
        return out


def integrate_streamlines(vf, seeds, pieces):
    """Arc-length RK2 streamlines rising from each seed until they run out
    of length or come within STREAM_CLEAR of a bubble surface. Returns one
    polyline per surviving seed, with its per-point alpha profile."""
    n_steps = int(STREAM_LEN / STREAM_STEP)
    centres = np.array([c for c, r, _ in pieces])
    radii = np.array([r for c, r, _ in pieces])

    def clear_of_bubbles(p):
        d = np.linalg.norm(p[:, None, :] - centres[None, :, :], axis=2)
        return (d - radii[None, :]).min(axis=1) > STREAM_CLEAR

    p = seeds.copy()
    # each line rises: pick the integration direction with upward velocity
    sign = np.where(vf.sample(p)[:, 2] >= 0, 1.0, -1.0)[:, None]
    alive = clear_of_bubbles(p)
    steps_alive = np.zeros(len(seeds), dtype=int)
    path = [p.copy()]
    for _ in range(n_steps):
        v1 = vf.sample(p)
        s1 = np.linalg.norm(v1, axis=1, keepdims=True)
        # a feeble relative flow gives direction noise, not a streamline
        alive &= s1[:, 0] > 5e-3
        v1 = np.where(s1 > 1e-12, v1 / np.maximum(s1, 1e-12), 0)
        mid = p + sign * 0.5 * STREAM_STEP * v1
        v2 = vf.sample(mid)
        s2 = np.linalg.norm(v2, axis=1, keepdims=True)
        v2 = np.where(s2 > 1e-12, v2 / np.maximum(s2, 1e-12), 0)
        p = p + np.where(alive[:, None], sign * STREAM_STEP * v2, 0)
        alive &= clear_of_bubbles(p)
        steps_alive += alive
        path.append(p.copy())
    path = np.array(path)                      # [n_steps+1, M, 3]
    kern = np.ones(5) / 5
    lines = []
    for m in range(len(seeds)):
        n = steps_alive[m] + 1
        if n < 8:
            continue
        pts = path[:n, m]
        pts = np.column_stack([np.convolve(pts[:, ax], kern, mode='valid')
                               for ax in range(3)])
        # soft birth at the bottom, soft death where the line stops
        ramp = np.linspace(0, 1, len(pts))
        alpha = np.minimum(np.minimum(ramp / 0.18, 1.0),
                           np.minimum((1.0 - ramp) / 0.30, 1.0))
        lines.append((pts, alpha))
    return lines


class Tracker:
    """Keep bubble identity across frames: the solver renumbers connected
    components (and breakup creates new ones), so each frame's components are
    matched to the previous frame by centroid distance and inherit the color
    of their closest ancestor.

    Unwrapping is done per mesh-connected piece, never per component id: a
    piece is always far smaller than half a domain period, so wrapping its
    nodes around its own circular centroid can never tear a triangle apart —
    which per-component wrapping did whenever a fragment sat exactly on the
    fold. Each piece is then translated by whole periods to sit next to its
    component's tracked position, keeping trajectories continuous across
    periodic boundaries."""

    def __init__(self):
        self.prev = []          # list of (centroid[3] unwrapped, color index)

    def process(self, nodes, tris, compo_of_tri):
        """Returns (unwrapped nodes, per-triangle color index, pieces),
        pieces being a list of (centroid, bounding radius, color index).
        Every piece is matched to its own previous-frame position over the
        periodic images, so no piece can ever jump a period between frames."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        dom = np.array(DOM)
        n = len(nodes)
        edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]],
                                tris[:, [2, 0]]])
        adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                         shape=(n, n))
        _, piece_of_node = connected_components(adj, directed=False)
        comp_of_node = np.zeros(n, dtype=int)
        comp_of_node[tris.ravel()] = np.repeat(compo_of_tri.astype(int), 3)

        out = nodes.copy()
        tri_piece = piece_of_node[tris[:, 0]]
        tri_color = np.zeros(len(tris), dtype=int)
        entries, pieces = [], []
        for p in np.unique(piece_of_node[tris.ravel()]):
            sel = piece_of_node == p
            # wrap the piece contiguously around its own circular centroid
            # (a piece is always far smaller than half a period)
            centre = np.array([circular_centre(out[sel, ax], DOM[ax]) % dom[ax]
                               for ax in range(3)])
            for ax in range(3):
                out[sel, ax] = unwrap_component(out[sel, ax], centre[ax],
                                                DOM[ax])
            cen = out[sel].mean(axis=0)
            if self.prev:
                # this piece's own nearest ancestor over periodic images:
                # a newborn fragment lands on its parent
                best = (np.inf, None, None)
                for pc, pcol in self.prev:
                    img = cen - np.round((cen - pc) / dom) * dom
                    d = np.linalg.norm(img - pc)
                    if d < best[0]:
                        best = (d, img, pcol)
                out[sel] += best[1] - cen
                cen, col = best[1], best[2]
            else:
                col = int(comp_of_node[sel][0])
            radius = np.linalg.norm(out[sel] - cen, axis=1).max()
            entries.append((cen, col))
            pieces.append((cen, radius, col))
            tri_color[tri_piece == p] = col
        self.prev = entries
        return out, tri_color, pieces


def make_background(global_bounds, t_end=0.0):
    """The drifting water frame: grid-plane data and micro-bubble layers.
    Every z extent is widened by that element's drift span so the field
    covers each frame of the playback at the same density."""
    rng = np.random.default_rng(BG_SEED)
    (x0, x1), _, (z0, z1) = global_bounds
    bg = {'x': (x0 - 0.4, x1 + 0.4)}
    if GRID_STRIDE > 0:
        # the grid is world-fixed: it is the static reference the camera
        # and the drifting bubble layers move against. Lines span the full
        # extent ever seen; the canvas clips whatever is off-screen.
        step = GRID_STRIDE * GRID_DX
        zlo, zhi = z0 - 0.4, z1 + 0.4
        # lines sit on mesh planes: z = k*dx with k a multiple of the stride
        kk = np.arange(math.ceil(zlo / step), math.floor(zhi / step) + 1)
        bg['grid_z'] = kk * step
        bg['grid_zext'] = (zlo, zhi)
        # plain increasing indices, zero at the lowest drawn line (as if
        # the simulated box were this tall)
        bg['grid_k'] = (kk - kk.min()) * GRID_STRIDE
        bg['grid_x'] = np.arange(math.ceil(bg['x'][0] / step),
                                 math.floor(bg['x'][1] / step) + 1) * step
    layers = []
    for n, ylo, yhi, smin, smax, a, ascend, wob in BUBBLE_LAYERS:
        w = -ascend            # forward-time drift; playback runs t backwards
        zlo = z0 - 0.4 - max(0.0, w * t_end)
        zhi = z1 + 0.4 + max(0.0, -w * t_end)
        m = int(n * (zhi - zlo) / (z1 - z0 + 0.8))
        layers.append({
            'pts': np.column_stack([rng.uniform(*bg['x'], m),
                                    rng.uniform(ylo, yhi, m),
                                    rng.uniform(zlo, zhi, m)]),
            'size': rng.uniform(smin, smax, m),
            'alpha': a * rng.uniform(0.4, 1.0, m),
            'w': w,
            'wobble': wob,
            'period': rng.uniform(0.6, 1.3, m),
            'phase': rng.uniform(0, 2 * np.pi, m),
            'front': ylo < 0,
        })
    bg['layers'] = layers
    return bg


def draw_grid(ax, bg):
    if 'grid_z' not in bg:
        return
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    x0, x1 = bg['x']
    zlo, zhi = bg['grid_zext']
    segs = [[(x0, GRID_Y, zz), (x1, GRID_Y, zz)] for zz in bg['grid_z']]
    segs_v = [[(xx, GRID_Y, zlo), (xx, GRID_Y, zhi)] for xx in bg['grid_x']]
    ax.add_collection3d(Line3DCollection(
        segs_v, colors=(*BG_RGB, GRID_ALPHA * 0.45), linewidths=0.7))
    ax.add_collection3d(Line3DCollection(
        segs, colors=(*BG_RGB, GRID_ALPHA), linewidths=0.9))
    if GRID_LABELS:
        for zz, k in zip(bg['grid_z'][::2], bg['grid_k'][::2]):
            ax.text(x1 - 0.45, GRID_Y, zz + 0.01, f'k={k}',
                    color=BG_RGB, alpha=GRID_ALPHA * 1.6, fontsize=6,
                    ha='right', va='bottom', zdir='x')


def draw_layer(ax, lay, t):
    x = lay['pts'][:, 0] + lay['wobble'] * np.sin(
        2 * np.pi * t / lay['period'] + lay['phase'])
    z = lay['pts'][:, 2] + lay['w'] * t
    face = np.column_stack([np.tile(BG_RGB, (len(x), 1)),
                            lay['alpha'] * 0.25])
    edge = np.column_stack([np.tile(BG_RGB, (len(x), 1)), lay['alpha']])
    ax.scatter(x, lay['pts'][:, 1], z, s=lay['size'], facecolors=face,
               edgecolors=edge, marker='o', linewidths=0.7,
               depthshade=False)


def render_frame(nodes, tris, compo, colors, out, bounds, azim=None,
                 trails=(), streams=(), stream_fade=1.0, bg=None, t=0.0):
    azim = CAM_AZIM if azim is None else azim
    fig = plt.figure(figsize=(12.8, 6.4), facecolor=BG)
    ax = fig.add_subplot(projection='3d', facecolor=BG)
    if bg is not None:                   # the world behind the bubbles
        draw_grid(ax, bg)
        for lay in bg['layers']:
            if not lay['front']:
                draw_layer(ax, lay, t)
    # lines first: added before the surfaces, they always draw under them
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    for pts, aprof in streams:
        if len(pts) < 3:
            continue
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        alpha = STREAM_ALPHA * stream_fade * aprof[:len(segs)]
        cols = np.column_stack([np.tile(STREAM_RGB, (len(segs), 1)), alpha])
        ax.add_collection3d(Line3DCollection(segs, colors=cols,
                                             linewidths=STREAM_LW,
                                             capstyle='round'))
    for pts, rgb, a0 in trails:
        if len(pts) < 3:
            continue
        # smooth the centroid wobble (and fragment-shift kinks) away
        k = min(7, 2 * (len(pts) // 2) - 1)
        kern = np.ones(k) / k
        sm = np.column_stack([np.convolve(pts[:, ax_], kern, mode='valid')
                              for ax_ in range(3)])
        sm[0] = pts[0]                      # keep the bubble-end anchored
        segs = np.stack([sm[:-1], sm[1:]], axis=1)
        fade = np.linspace(a0, 0.0, len(segs)) ** 1.5
        cols = np.column_stack([np.tile(rgb, (len(segs), 1)), fade])
        ax.add_collection3d(Line3DCollection(
            segs, colors=cols, capstyle='round',
            linewidths=np.linspace(TRAIL_LW, 0.5, len(segs))))
    v = nodes[tris]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    lightdir = LIGHT / np.linalg.norm(LIGHT)
    shade = 0.55 + 0.45 * np.clip(n @ lightdir, 0, 1)
    face = colors[compo.astype(int) % len(colors)] * shade[:, None]
    # painter's ordering: draw far triangles first, along the camera axis
    el, az = np.deg2rad(CAM_ELEV), np.deg2rad(azim)
    campos = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                       np.sin(el)])
    order = np.argsort(v.mean(axis=1) @ campos)
    pc = Poly3DCollection(v[order], facecolors=face[order],
                          edgecolors='none', antialiased=False)
    ax.add_collection3d(pc)
    if bg is not None:                  # the few bubbles in front of it all
        for lay in bg['layers']:
            if lay['front']:
                draw_layer(ax, lay, t)
    (x0, x1), (y0, y1), (z0, z1) = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0 - 1.0, y1 + 1.0)   # widen y so perspective stays gentle
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0 + 2.0, z1 - z0))
    ax.set_proj_type('persp', focal_length=CAM_FOCAL)
    ax.view_init(elev=CAM_ELEV, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=-0.35, top=1.35)
    fig.savefig(out, dpi=RENDER_DPI, facecolor=BG)
    plt.close(fig)


def main():
    global DOM
    case = sys.argv[1] if len(sys.argv) > 1 else 'logo'
    t_max = float(sys.argv[2]) if len(sys.argv) > 2 else np.inf
    m = lata.Master('.', case)
    DOM = tuple(m.domain_size())
    print('domain from lata:', [round(d, 3) for d in DOM])
    colors = glyph_colors()
    tracker = Tracker()
    dumps = []      # every dump: trails need centroids beyond t_max too
    seen = 0
    for i in range(len(m.times)):
        if not m.has('SOMMETS', i, 'INTERFACES'):
            continue
        seen += 1
        if seen <= START_DUMP:
            continue
        nodes, tris = m.mesh(i)
        compo = m.field('COMPO_CONNEXE', i, 'INTERFACES')
        nodes, color_idx, pieces = tracker.process(nodes, tris, compo)
        dumps.append((m.times[i], nodes, tris, color_idx, pieces, i))
    shown = [d for d in dumps if d[0] <= t_max]
    t_end = shown[-1][0]
    # the uniform ascent: rendered positions = simulation + lift, the lift
    # growing as the playback advances (forward time decreasing), so the
    # whole swarm climbs past the world-fixed grid
    for t, nodes, tris_, ci_, pieces, mi_ in dumps:
        lift = ASCEND * (t_end - t)
        nodes[:, 2] += lift
        for cen, _r, _c in pieces:
            cen[2] += lift
    lo = np.min([d[1].min(axis=0) for d in shown], axis=0) - 0.05
    hi = np.max([d[1].max(axis=0) for d in shown], axis=0) + 0.05
    bounds = list(zip(lo, hi))
    print('bounds:', [(round(a, 2), round(b, 2)) for a, b in bounds])
    if CAM_FOLLOW:
        los = np.array([d[1].min(axis=0) for d in shown])
        his = np.array([d[1].max(axis=0) for d in shown])
        centres_c = (los + his) / 2
        sizes_c = his - los
        k = min(CAM_SMOOTH, len(shown)) | 1          # odd window
        kern = np.ones(k) / k
        pad = k // 2
        smooth = lambda a: np.column_stack(
            [np.convolve(np.pad(a[:, ax], pad, mode='edge'), kern,
                         mode='valid') for ax in range(3)])
        centres_c, sizes_c = smooth(centres_c), smooth(sizes_c)
        half = (1 + CAM_MARGIN) * sizes_c / 2
        # keep the output's 2:1 frame: widen whichever of x/z is short
        half[:, 0] = np.maximum(half[:, 0], 2 * half[:, 2])
        half[:, 2] = np.maximum(half[:, 2], half[:, 0] / 2)
        frame_bounds = [list(zip(c - h, c + h))
                        for c, h in zip(centres_c, half)]
    frames = []
    bg = make_background(bounds, t_end=t_end)
    (bx0, bx1), _, (bz0, bz1) = bounds
    for idx, (t, nodes, tris, compo, pieces, mi) in enumerate(shown):
        fade = min(1.0, t / TRAIL_FADE_T) if TRAIL_FADE_T > 0 else 1.0
        # the trail of a bubble is where it goes next in forward time —
        # i.e. the path it just rose along in the reversed playback.
        # (piece indexing across dumps is approximate; trails are off by
        # default, superseded by the streamline curtain)
        trails = []
        if TRAIL_T > 0 and fade > 0.02:
            future = [d[4] for d in dumps[idx:]
                      if d[0] <= t + TRAIL_T]
            for e in range(len(pieces)):
                pts = np.array([f[e][0] for f in future if e < len(f)])
                trails.append((pts, colors[pieces[e][2]],
                               TRAIL_ALPHA * fade))
        # the streamline curtain: fixed seeds along the frame bottom, rising
        # through the falling-frame flow until just under the bubbles
        streams = []
        if (STREAM_COLS > 0 and fade > 0.02
                and m.has('CELL_VELOCITY_X', mi)):
            vf = VelocityField(m, mi)
            xs = np.linspace(bx0 + 0.06, bx1 - 0.06, STREAM_COLS)
            # deterministic golden-ratio stagger in depth
            ys = 0.32 + 0.55 * ((np.arange(STREAM_COLS) * 0.618) % 1.0 - 0.5)
            seeds = np.column_stack([xs, ys, np.full(STREAM_COLS, bz0 + 0.03)])
            streams = integrate_streamlines(vf, seeds, pieces)
        # forward time t_end -> 0 is playback start -> end: sweep to CAM_AZIM
        azim = CAM_AZIM + CAM_AZIM_SWEEP * (t / t_end)
        out = f'frame_{len(frames):03d}.png'
        render_frame(nodes, tris, compo, colors, out,
                     frame_bounds[idx] if CAM_FOLLOW else bounds,
                     azim=azim, trails=trails, streams=streams,
                     stream_fade=fade, bg=bg, t=t)
        frames.append(out)
        print(f'{out}  t={t:.3f}  azim={azim:.1f}  tris={len(tris)}')
    if len(frames) < 2:
        print('not enough frames for a gif')
        return
    # reverse playback: dispersed bubbles assemble into the logo
    imgs = [Image.open(f).resize(GIF_SIZE, Image.LANCZOS)
            for f in reversed(frames)]
    durations = [FRAME_MS] * len(imgs)
    durations[0] = 600            # short hold on the bubble cloud
    durations[-1] = 2500          # hold the assembled logo
    imgs[0].save(f'{case}_reverse.gif', save_all=True, append_images=imgs[1:],
                 duration=durations, loop=0, optimize=True)
    print(f'wrote {case}_reverse.gif ({len(imgs)} frames)')
    # the MP4 keeps the frames' native resolution
    import shutil
    import subprocess
    if shutil.which('ffmpeg'):
        cwd = os.getcwd()
        with open('frames_rev.txt', 'w') as f:
            for fr in reversed(frames):
                f.write(f"file '{cwd}/{fr}'\nduration {FRAME_MS / 1000}\n")
            f.write(f"file '{cwd}/{frames[0]}'\nduration 2.5\n")
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat',
                        '-safe', '0', '-i', 'frames_rev.txt',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', str(MP4_CRF), '-r', '25',
                        f'{case}_reverse.mp4'], check=True)
        print(f'wrote {case}_reverse.mp4')
    else:
        print('ffmpeg not found: no mp4 written')


if __name__ == '__main__':
    main()

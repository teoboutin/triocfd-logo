"""Build the initial front-tracking interface for the TrioCFD logo animation.

Each glyph of the logo becomes one closed triangulated surface (one
"bubble"): the 2D glyph is extruded along y, the blocky voxel surface is
extracted watertight, then Taubin-smoothed so the FT curvature computation
does not see staircase corners.

Most glyphs are connected components of the alpha mask (solid letters).
Two are rebuilt instead of taken from the opaque fill, to match the actual
logo artwork:
  - the o/C circle is really a C ring around a droplet (plus a 1.5 px inner
    circle, too thin to be a bubble, discarded). Extracting the ring from
    the drawn ink leaves a spur where the artwork's thin closing arc joins
    it, so the C is authored instead: an annulus fitted to the drawn ring
    (centre, mid-radius 20 px, thickness 6 px) with a 50 degree opening
    toward the upper-right notch and rounded end caps. The droplet is still
    extracted from the ink (it is clean).
  - the nabla is a triangular border around white fill: kept as the border
    ring alone. That surface has genus 1 (a torus bubble).

Each glyph is offset a few pixels along y so the logo has depth parallax
under an inclined camera: T farthest, nabla closest. The droplet carries
the C's offset, staying inside its ring.

With ADD_O_RING, a thin closed O ring is added behind the C (deeper in y,
slightly larger radius so its rim peeks past the C's silhouette): the
artwork's circle is an O and a C at once, and the third dimension lets the
two coexist as separate bubbles.

Output: init.lata (+ .nodes/.elem/.connex_compo) in the ASCII LATA
interface format the IJK FT reader expects (F_INDEXING, outward normals),
and preview.png to eyeball the result.

The logo has a fixed physical size (1 px = 0.01 m); the domain around it is
variable — the glyphs are centred in whatever box the caller passes (the
pipeline driver aligns it with the deck it generates).

CLI: python3 make_letters.py [--domain LX LY LZ] [--out DIR]
"""
import argparse
import os
import numpy as np
from PIL import Image
from scipy import ndimage, sparse

PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tcfd.png')

PX = 0.01            # metres per pixel
DOM = (2.56, 0.64, 1.28)     # default domain, overridable per call
THICK_PX = 12        # extrusion thickness along y, in pixels
ERODE_PX = 1         # shrink solid glyphs to widen inter-glyph gaps
INK_THR = 190        # min(R,G,B) below this = drawn ink, not white fill
TAUBIN_ITERS = 30    # smoothing passes (pairs of lambda/mu steps)

# The authored C: an annulus with an opening toward the upper-right notch of
# the artwork, rounded end caps. Fitted to the drawn ring (see docstring).
# Image coords: x right, y down; angles in math orientation (y up).
C_GAP_DIR = np.deg2rad(45.0)     # up-right
C_GAP_HALF = np.deg2rad(25.0)
C_R_MID = 20.0
C_THICK = 6.0

# Per-glyph y offset in pixels, by alpha-component left-to-right order
# (T r swoosh i o/C F nabla). Camera looks from -y, so negative = closer to
# the viewer: the logo recedes from nabla to T. C and droplet share one
# offset so the droplet stays inside the ring.
Y_STAGGER = {0: +4, 1: +2, 2: -2, 3: +1, 4: 0, 5: -2, 6: -4}

# The swoosh's arrowhead barbs are 2-3 px: eroding them makes the arrow melt
# into a club within the first few steps. Pixels right of this column keep
# their full drawn size (the tail is still eroded like every solid glyph).
ARROW_X_MIN = 178

# The O behind the C: a thin closed ring, deeper in y. y offset 16 px puts
# a 4 px gap (2 cells) behind the C's 12 px slab; the 8 px extrusion keeps
# it inside the periodic y domain with margin.
ADD_O_RING = False
O_R_MID = 24.5
O_THICK = 4.0
O_Y_OFF = 16
O_EXTRUDE = 8


def _ink_subglyphs(comp_mask, rgb_min):
    """Split one alpha component into its drawn-ink pieces."""
    ink = comp_mask & (rgb_min < INK_THR)
    lab, n = ndimage.label(ink)
    return [(lab == i) for i in range(1, n + 1)]


def _author_c(ring, shape, scale=1):
    """Draw the C as a clean annulus sector with round caps, centred on the
    drawn ring's filled-disk centroid."""
    filled = ndimage.binary_fill_holes(ring)
    ys, xs = np.where(filled)
    cy, cx = ys.mean(), xs.mean()
    r_mid, half_t = C_R_MID * scale, C_THICK * scale / 2
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    r = np.hypot(xx - cx, yy - cy)
    ang = np.arctan2(-(yy - cy), xx - cx)
    diff = np.abs(np.angle(np.exp(1j * (ang - C_GAP_DIR))))
    out = (np.abs(r - r_mid) <= half_t) & (diff >= C_GAP_HALF)
    for end in (C_GAP_DIR + C_GAP_HALF, C_GAP_DIR - C_GAP_HALF):
        ex, ey = cx + r_mid * np.cos(end), cy - r_mid * np.sin(end)
        out |= np.hypot(xx - ex, yy - ey) <= half_t
    lab, n = ndimage.label(out)
    assert n == 1, f'authored C should be one piece, got {n}'
    return out


def _author_o(ring, shape, scale=1):
    """Draw the O as a thin closed annulus, concentric with the C."""
    filled = ndimage.binary_fill_holes(ring)
    ys, xs = np.where(filled)
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    r = np.hypot(xx - cx, yy - cy)
    return np.abs(r - O_R_MID * scale) <= O_THICK * scale / 2


def load_glyphs(scale=1):
    """The glyphs, left to right, as dicts {mask, y_off, thick, shade}.
    Solid letters are eroded ERODE_PX; ink-extracted glyphs (C ring, droplet,
    nabla ring) are kept unshrunk — they are thin, and their gaps to the
    neighbours are already wide. shade darkens a glyph's display colour.

    scale supersamples the artwork: the masks are extracted on a grid of
    scale x scale sub-pixels (smooth LANCZOS upscale of the RGBA image), so
    the voxel size — and with it the staircase the initial interface
    carries — shrinks by that factor. All pixel-based constants scale."""
    img = Image.open(PNG).convert('RGBA')
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale),
                         Image.LANCZOS)
    im = np.array(img)
    mask = im[..., 3] > 128
    rgb_min = im[..., :3].astype(int).min(axis=2)
    lab, n = ndimage.label(mask)
    comps = sorted((lab == i for i in range(1, n + 1)),
                   key=lambda m: np.where(m)[1].min())

    def glyph(m, idx, **kw):
        g = dict(mask=m, y_off=Y_STAGGER[idx] * scale,
                 thick=THICK_PX * scale, shade=1.0)
        g.update(kw)
        return g

    # left-to-right alpha components: T r swoosh i o F nabla
    glyphs = []
    for idx, m in enumerate(comps):
        if idx == 4:      # the o/C circle: C ring + droplet, inner circle out
            subs = _ink_subglyphs(m, rgb_min)
            subs.sort(key=lambda s: s.sum(), reverse=True)
            ring, rest = subs[0], subs[1:]
            glyphs.append(glyph(_author_c(ring, m.shape, scale), idx))
            for s in rest:
                hole = ndimage.binary_fill_holes(s).sum() / s.sum()
                if hole < 2.0:                # the droplet; rings are > 2
                    glyphs.append(glyph(s, idx))
            if ADD_O_RING:
                glyphs.append(glyph(_author_o(ring, m.shape, scale), idx,
                                    y_off=(Y_STAGGER[idx] + O_Y_OFF) * scale,
                                    thick=O_EXTRUDE * scale, shade=0.8))
        elif idx == 6:    # the nabla: border ring only (genus 1)
            subs = _ink_subglyphs(m, rgb_min)
            glyphs.append(glyph(max(subs, key=lambda s: s.sum()), idx))
        else:
            er = ndimage.binary_erosion(m, iterations=ERODE_PX * scale)
            if idx == 2:  # swoosh: keep the arrowhead at full drawn size
                cols = np.arange(m.shape[1])[None, :]
                er = er | (m & (cols >= ARROW_X_MIN * scale))
            glyphs.append(glyph(er, idx))
    glyphs.sort(key=lambda g: np.where(g['mask'])[1].min())
    return glyphs, im


def glyph_colors():
    """One display colour per glyph (same ordering as load_glyphs), the
    median of its darkest 30 % of opaque pixels, kept readable on a dark
    ground, times the glyph's shade."""
    glyphs, im = load_glyphs()
    opaque = im[..., 3] > 128
    cols = []
    for g in glyphs:
        m = g['mask'] & opaque
        if not m.any():
            m = g['mask']
        px = im[m][:, :3].astype(float)
        lum = px.mean(axis=1)
        dark = px[lum <= np.percentile(lum, 30)]
        rgb = np.median(dark, axis=0) / 255.0
        mean = rgb.mean()
        if mean < 0.35:
            rgb = rgb + (0.35 - mean)
        elif mean > 0.75:
            rgb = rgb * (0.75 / mean)
        cols.append(np.clip(rgb * g['shade'], 0, 1))
    return np.array(cols)


def voxel_surface(mask2d, thick=THICK_PX):
    """Watertight triangulated boundary of the extruded glyph.

    Voxels: (col i, layer j, row k) with j the extrusion direction.
    Returns (points [n,3] lattice coords, triangles [m,3] 0-based),
    triangles wound so normals point outward.
    """
    ny, nx = mask2d.shape
    vox = np.zeros((nx, int(thick), ny), dtype=bool)
    vox[:, :, :] = mask2d.T[:, None, :]

    pad = np.pad(vox, 1)
    quads = []  # each: (4 corner lattice points, outward axis, sign)
    for axis in range(3):
        for sign in (+1, -1):
            shifted = np.roll(pad, -sign, axis=axis)
            faces = np.argwhere(pad & ~shifted) - 1  # voxels with a boundary face
            if len(faces) == 0:
                continue
            # corner offsets of that face, wound so the normal points along sign*axis
            base = np.zeros((4, 3), dtype=int)
            u, v = (axis + 1) % 3, (axis + 2) % 3
            if sign > 0:
                base[:, axis] = 1
                base[1, u] = 1
                base[2, u] = 1
                base[2, v] = 1
                base[3, v] = 1
            else:
                base[1, v] = 1
                base[2, u] = 1
                base[2, v] = 1
                base[3, u] = 1
            quads.append(faces[:, None, :] + base[None, :, :])
    quads = np.concatenate(quads)                      # [nq, 4, 3]
    corners = quads.reshape(-1, 3)
    points, inverse = np.unique(corners, axis=0, return_inverse=True)
    q = inverse.reshape(-1, 4)
    tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return points.astype(float), tris


def taubin_smooth(points, tris, iters=TAUBIN_ITERS, lam=0.5, mu=-0.53):
    """Taubin lambda/mu smoothing: rounds the staircase without shrinking."""
    n = len(points)
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    a = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                          shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(float)
    deg = np.asarray(a.sum(axis=1)).ravel()
    p = points.copy()
    for _ in range(iters):
        for step in (lam, mu):
            p += step * (a @ p / deg[:, None] - p)
    return p


def mesh_volume(points, tris):
    v0, v1, v2 = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return np.einsum('ij,ij->', v0, np.cross(v1, v2)) / 6.0


def build(dom=DOM, scale=1):
    """scale supersamples the initial interface: voxel size PX/scale."""
    glyphs, im = load_glyphs(scale)
    px = PX / scale
    img_h, img_w = im.shape[:2]
    x0 = (dom[0] - img_w * px) / 2.0          # centre the logo in x
    y0 = (dom[1] - THICK_PX * PX) / 2.0       # centre the slab in y
    z0 = (dom[2] - img_h * px) / 2.0          # centre in z
    assert x0 >= 0 and y0 >= 0 and z0 >= 0, \
        f'domain {dom} too small for the logo ({img_w * px} x ' \
        f'{THICK_PX * PX} x {img_h * px} m)'
    # the staircase wavelength shrinks with the voxels: smoothing the same
    # physical scale needs iterations growing like scale^2
    iters = min(TAUBIN_ITERS * scale * scale, 300)
    all_nodes, all_tris, all_compo = [], [], []
    offset = 0
    for ci, g in enumerate(glyphs):
        pts, tris = voxel_surface(g['mask'], g['thick'])
        pts = taubin_smooth(pts, tris, iters=iters)
        # lattice -> physical: x = column, y = layer + stagger, z = flipped row
        phys = np.empty_like(pts)
        phys[:, 0] = x0 + pts[:, 0] * px
        phys[:, 1] = y0 + (pts[:, 1] + g['y_off']) * px
        phys[:, 2] = z0 + (img_h - pts[:, 2]) * px
        # z-flip mirrors the mesh: rewind triangles to keep normals outward
        tris = tris[:, ::-1]
        vol = mesh_volume(phys, tris)
        r_eq = (3 * abs(vol) / (4 * np.pi)) ** (1 / 3)
        print(f'compo {ci}: {len(phys):6d} nodes {len(tris):6d} tris '
              f'volume {vol:.6e} m3  r_eq {r_eq:.4f} m')
        assert vol > 0, 'normals must point outward'
        all_nodes.append(phys)
        all_tris.append(tris + 1 + offset)    # F_INDEXING: 1-based
        all_compo.append(np.full(len(tris), ci, dtype=int))
        offset += len(phys)
    return (np.concatenate(all_nodes), np.concatenate(all_tris),
            np.concatenate(all_compo))


def write_lata(path, nodes, elements, compo):
    base = path.split('/')[-1]
    np.savetxt(path + '.nodes', nodes, fmt='%.9g')
    np.savetxt(path + '.elem', elements, fmt='%d')
    np.savetxt(path + '.connex_compo', compo, fmt='%d')
    with open(path, 'w') as f:
        f.write('LATA_V2.1\nlogo_animation\nTrio_U\n')
        f.write('Format ASCII,F_INDEXING,C_ORDERING,F_MARKERS_NO,INT32,REAL32\n')
        f.write('TEMPS 0\n')
        f.write('Geom INTERFACES type_elem=TRIANGLE_3D\n')
        f.write(f'Champ SOMMETS  {base}.nodes geometrie=INTERFACES '
                f'size={len(nodes)} composantes=3\n')
        f.write(f'Champ ELEMENTS {base}.elem geometrie=INTERFACES '
                f'size={len(elements)} composantes=3 FORMAT=INT32\n')
        f.write(f'Champ COMPO_CONNEXE {base}.connex_compo geometrie=INTERFACES '
                f'size={len(elements)} composantes=1 '
                f'FORMAT=INT32,NO_INDEXING localisation=ELEM\n')
        f.write('FIN\n')


def preview(nodes, elements, compo, out='preview.png', dom=DOM):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(projection='3d')
    tri = nodes[elements - 1]
    pc = Poly3DCollection(tri, facecolors=plt.cm.tab10(compo % 10),
                          edgecolors='none', alpha=0.9)
    ax.add_collection3d(pc)
    ax.set_xlim(0, dom[0]); ax.set_ylim(-0.9, 1.5); ax.set_zlim(0, dom[2])
    ax.set_box_aspect((dom[0], 2.4, dom[2]))
    ax.view_init(elev=8, azim=-88)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', nargs=3, type=float, default=list(DOM),
                    metavar=('LX', 'LY', 'LZ'))
    ap.add_argument('--supersample', type=int, default=1,
                    help='interface resolution factor (voxel = 1cm / S)')
    ap.add_argument('--out', default='.')
    args = ap.parse_args()
    dom = tuple(args.domain)
    nodes, elements, compo = build(dom, scale=args.supersample)
    print(f'total: {len(nodes)} nodes, {len(elements)} triangles, '
          f'{compo.max() + 1} bubbles')
    write_lata(os.path.join(args.out, 'init.lata'), nodes, elements, compo)
    preview(nodes, elements, compo, out=os.path.join(args.out, 'preview.png'),
            dom=dom)

#!/usr/bin/env python3
"""One-command pipeline for the TrioCFD logo animation.

Generates the interface geometry and the deck from their templates, runs the
simulation, renders the frames and assembles the reversed GIF and MP4:

    python3 pipeline.py --out runs/demo
    python3 pipeline.py --out runs/hires --dx 0.01 --steps 2000 --nproc 4 2 2

The visual quality is driven by the resolution knobs:
  --dx           fluid cell size (default 0.02 m -> 128x32x64 cells).
                 The interface remeshing targets edges of 0.5*dx, so the
                 simulated interface refines with the grid automatically.
                 Halving dx costs ~16x (8x cells, CFL halves the step).
  --supersample  resolution of the *initial* letter surfaces: the artwork is
                 voxelized at 1cm/S. Defaults to 0.02/dx rounded, so a finer
                 grid automatically starts from finer letters.
  --steps        nb_pas_dt_max; more steps = longer fall, more dispersal.

Needs the TrioCFD environment (source env_TrioCFD.sh) for the simulation
step; --render-only reruns the rendering of an existing case without it.

The logo has a fixed physical size (about 2.0 x 0.12 x 0.9 m); --domain
changes the box around it, letters centred. Camera, colors and output
quality live as knobs at the top of render.py; physics not exposed here
lives in logo.data.template.
"""
import argparse
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = 'logo'


def fail(msg):
    sys.exit(f'pipeline: {msg}')


def generate(args, out):
    sys.path.insert(0, HERE)
    import make_letters
    dom = tuple(args.domain)
    scale = args.supersample or max(1, round(0.02 / args.dx))
    print(f'interface supersampling: {scale} (voxel {1.0 / scale:.2f} cm)')
    nodes, elements, compo = make_letters.build(dom, scale=scale)
    make_letters.write_lata(os.path.join(out, 'init.lata'),
                            nodes, elements, compo)
    make_letters.preview(nodes, elements, compo,
                         out=os.path.join(out, 'preview.png'), dom=dom)

    nb = [round(L / args.dx) for L in dom]
    for L, n, np_, ax in zip(dom, nb, args.nproc, 'xyz'):
        if abs(n * args.dx - L) > 1e-9:
            fail(f'L{ax} = {L} is not a multiple of dx = {args.dx}')
        if n % (4 * np_):
            fail(f'nbelem_{ax} = {n} must be divisible by 4*nproc_{ax} '
                 f'= {4 * np_} (two multigrid coarsenings per rank)')
    subst = {
        '@NBI@': nb[0], '@NBJ@': nb[1], '@NBK@': nb[2],
        '@LX@': dom[0], '@LY@': dom[1], '@LZ@': dom[2],
        '@NPI@': args.nproc[0], '@NPJ@': args.nproc[1],
        '@NPK@': args.nproc[2],
        '@STEPS@': args.steps, '@DTPOST@': args.dt_post,
        # the swirl axis: vertical, off-centre by the same offset as the
        # original 2.56 m domain (0.38 m left of centre), in the slab plane
        '@XAXIS@': round(dom[0] / 2 - 0.38, 6),
        '@YMID@': round(dom[1] / 2, 6),
    }
    with open(os.path.join(HERE, 'logo.data.template')) as f:
        deck = f.read()
    for key, val in subst.items():
        deck = deck.replace(key, str(val))
    if '@' in deck:
        fail('unsubstituted placeholder left in the deck')
    with open(os.path.join(out, f'{CASE}.data'), 'w') as f:
        f.write(deck)
    print(f'generated {CASE}.data ({nb[0]}x{nb[1]}x{nb[2]} cells) '
          f'and init.lata in {out}')


def simulate(args, out):
    exe = args.executable or os.environ.get('exec', '')
    if not exe:
        fail('no executable: pass --exec or source the TrioCFD environment')
    nprocs = args.nproc[0] * args.nproc[1] * args.nproc[2]
    cmd = f'exec="{exe}" trust {CASE}.data {nprocs}'
    print(f'running: {cmd}   (log: {out}/run.log)')
    with open(os.path.join(out, 'run.log'), 'w') as log:
        r = subprocess.run(cmd, shell=True, cwd=out, stdout=log,
                           stderr=subprocess.STDOUT)
    # the exit code alone is not trusted: the end-of-run save must exist
    if r.returncode != 0 or not os.path.exists(
            os.path.join(out, f'{CASE}.sauv')):
        fail(f'simulation failed (exit {r.returncode}) — see {out}/run.log')
    print('simulation finished')


def render(args, out):
    cmd = [sys.executable, os.path.join(HERE, 'render.py'), CASE]
    if args.render_tmax is not None:
        cmd.append(str(args.render_tmax))
    r = subprocess.run(cmd, cwd=out)
    if r.returncode != 0:
        fail('rendering failed')
    print(f'movie: {out}/{CASE}_reverse.gif and .mp4')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='case directory (created)')
    ap.add_argument('--domain', nargs=3, type=float,
                    default=[2.56, 0.64, 1.28], metavar=('LX', 'LY', 'LZ'))
    ap.add_argument('--dx', type=float, default=0.02,
                    help='cubic cell size (default 0.02 m); the main quality knob')
    ap.add_argument('--supersample', type=int, default=0,
                    help='initial interface resolution factor '
                         '(default: 0.02/dx rounded)')
    ap.add_argument('--steps', type=int, default=750,
                    help='nb_pas_dt_max (default 750, about 3 s simulated)')
    ap.add_argument('--nproc', nargs=3, type=int, default=[2, 2, 2],
                    metavar=('NI', 'NJ', 'NK'))
    ap.add_argument('--dt-post', type=int, default=5,
                    help='time steps between dumps (default 5)')
    ap.add_argument('--render-tmax', type=float, default=None,
                    help='render dumps up to this time only (default: all)')
    ap.add_argument('--exec', dest='executable', default=None,
                    help='TrioCFD binary (default: $exec from the environment)')
    ap.add_argument('--render-only', action='store_true',
                    help='skip generation and simulation, render an existing case')
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if not args.render_only:
        generate(args, out)
        simulate(args, out)
    render(args, out)


if __name__ == '__main__':
    main()

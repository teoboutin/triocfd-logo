# TrioCFD logo animation

The TrioCFD logo dissolving into rising bubbles — simulated with TrioCFD
itself (IJK front tracking), then played in reverse so the bubbles rise from
below, swirl, and assemble into the logo.

Each glyph of the logo is one closed triangulated surface (a "bubble") of
the light phase. The bubbles relax toward round shapes under surface
tension, shed a few fragments, and the whole periodic box free-falls under
gravity while an off-centre swirl and staggered kicks disperse them. The
movie is the dump sequence rendered backwards, with a camera that follows
the swarm and a world-fixed field of suspended particles as the depth
reference.

## Quickstart

Everything runs from one command, in a sourced TrioCFD environment
(`source env_TrioCFD.sh` in the TrioCFD checkout):

    python3 pipeline.py --out runs/demo

That generates the interface geometry (`init.lata`) and the deck from
`logo.data.template`, runs the simulation (8 MPI ranks by default), renders
every dump and assembles `logo_reverse.gif` and `logo_reverse.mp4` in the
case directory. `--render-only` redoes just the rendering of an existing
case (no TrioCFD environment needed).

## Resolution — the quality knobs

    python3 pipeline.py --out runs/hires --dx 0.01 --steps 2000 --nproc 4 2 2

- `--dx` (default 0.02 m): the fluid cell size, and through it the simulated
  interface resolution — the remeshing targets triangle edges of 0.5*dx.
  Halving dx costs roughly 16x (8x cells, CFL halves the time step).
- `--supersample` (default 0.02/dx, rounded): the resolution of the initial
  letter surfaces — the artwork is voxelized at 1 cm / S. It follows --dx
  automatically; force it higher for crisper letters on a coarse grid.
- `--steps`: `nb_pas_dt_max`. More steps = deeper fall and more dispersal.
  With the default velocities, dt is about 2.7e-3 s, so 750 steps simulate
  about 2 s.
- `--domain LX LY LZ` (default 2.56 0.64 1.28): the periodic box. The logo
  has a fixed physical size (about 2.0 x 0.12 x 0.9 m), centred in it.
- `--nproc NI NJ NK`: MPI decomposition; each direction's cell count must be
  divisible by 4*nproc (two multigrid coarsenings per rank) — the driver
  checks.
- Render quality: `RENDER_DPI` (150 = 1920x960 frames), `GIF_SIZE`,
  `MP4_CRF` at the top of `render.py`, next to the camera knobs
  (`CAM_ELEV/AZIM/FOCAL`, azimuth sweep, follow) and the background field
  (`BG_*`).

## The files

- `tcfd.png` — the logo artwork; everything derives from it.
- `make_letters.py` — glyph masks -> closed watertight surfaces ->
  `init.lata` (ASCII LATA, outward normals, one `COMPO_CONNEXE` per glyph).
  Solid letters come from the alpha mask (eroded 1 px for clearance); the
  C is authored (annulus with a 50-degree opening, round caps — the drawn
  ring reads closed because the swoosh's shadow bridges its gap); the
  droplet is extracted from the drawn ink; the nabla keeps only its border,
  a genus-1 torus bubble the FT machinery handles fine. The swoosh's
  arrowhead skips the erosion so its 2-3 px barbs survive. `ADD_O_RING`
  optionally puts a thin closed O behind the C (unused by default).
- `logo.data.template` — the deck: `Probleme_FTD_IJK`, RK3, fully periodic,
  rho 100/1000, sigma 1, g 0.4, inter-bubble repulsion, remeshing at
  0.5*dx with little smoothing (the default remesh smoothing erased the
  arrowhead within 5 steps), initial velocity = off-centre rigid swirl
  about a vertical axis + dispersal kicks, `VELOCITY elem` dumped for the
  renderer.
- `pipeline.py` — the driver (see Quickstart).
- `render.py` — reads the LATA dumps (`lata.py`, a reader borrowed from a
  TrioCFD validation form), tracks every mesh-connected piece across frames
  (colors stay stable through breakup; periodic unwrapping is per piece, so
  no bubble can ever jump a period between frames), renders shaded frames
  with a camera that follows the smoothed cloud bounding box, and assembles
  the reversed GIF + MP4. Optional layers behind knobs: Lagrangian trails,
  velocity streamlines (both off — judged not good-looking), the
  marine-snow background (on).

## Physics notes (the non-obvious ones)

- In a fully periodic box, a uniform gravity cannot be balanced by the
  periodic pressure: the whole mixture free-falls, and there is no relative
  buoyancy in the falling frame. That free fall IS the animation — reversed,
  everything rises together. (`compute_force_init` would cancel it and give
  true buoyant rise instead; tried, and the wake capture made bubbles
  collide.)
- The renderer's streamline option must subtract the volume-mean velocity
  for the same reason, or it draws the fall instead of the flow.
- A rigid rotation preserves mutual distances: the swirl gives the carousel
  look but zero dispersal. Dispersal comes from the differential kicks in
  the initial velocity expressions.
- Bubbles kiss (2-3 mm at closest, held by `portee_force_repulsion` /
  `delta_p_max_repulsion`) but front tracking cannot merge them.
- Fragments shed by the thin rings keep their parent's `COMPO_CONNEXE`, so
  they keep the parent's color and re-merge into it in reverse.

## Warnings

- Do not `rm *.lata*` in a case directory: that deletes `init.lata`.
- Wall-clock on a 16-core desktop, 8 ranks, defaults: about 10 min for
  500 steps. Scale expectations with (cells) x (1/dx) when refining.

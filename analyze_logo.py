"""Inspect the TrioCFD logo: connected components of the opaque mask."""
import numpy as np
from PIL import Image
from scipy import ndimage

im = Image.open('/volatile/catA/tb266682/triocfd_workspace/tcfd.png').convert('RGBA')
a = np.array(im)
print('shape', a.shape)
alpha = a[..., 3]
mask = alpha > 128
print('opaque fraction', mask.mean())

lab, n = ndimage.label(mask)
print('components:', n)
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    filled = ndimage.binary_fill_holes(lab == i)
    holes = filled.sum() - (lab == i).sum()
    # mean color of the component
    rgb = a[lab == i][:, :3].mean(axis=0).astype(int)
    print(f'comp {i}: npix={len(xs):5d} bbox x[{xs.min()},{xs.max()}] '
          f'y[{ys.min()},{ys.max()}] hole_pix={holes} mean_rgb={tuple(rgb)}')

# save a visualization: each component in a distinct color, upscaled x4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(np.where(lab == 0, np.nan, lab), cmap='tab20', interpolation='nearest')
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    ax.text(xs.mean(), ys.mean(), str(i), color='k', fontsize=14, ha='center')
ax.set_title('connected components of alpha>128')
fig.savefig('components.png', dpi=110, bbox_inches='tight')
print('wrote components.png')

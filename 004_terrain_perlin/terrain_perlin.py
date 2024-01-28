import pyvista as pv
import imageio
from tqdm.auto import tqdm

freq = (1, 1, 1)
noise = pv.perlin_noise(1, freq, (0, 0, 0))
grid = pv.sample_function(noise, [0, 3.0, -0, 1.0, 0, 1.0], dim=(120, 40, 40))
out = grid.threshold(0.02)

mn, mx = [out['scalars'].min(), out['scalars'].max()]
clim = (mn, mx * 1.8)

pl = pv.Plotter(off_screen=True)
pl.add_mesh(out)

camera = pl.camera

screenshots = []
num_iters = 60
for i in tqdm(range(1, num_iters + 1)):

    freq = (1, 1, 1)
    noise = pv.perlin_noise(1, freq, (i/num_iters * 2, 0, 0))
    grid = pv.sample_function(noise, [0, 3.0, -0, 1.0, 0, 1.0], dim=(120, 40, 40))
    out = grid.threshold(0.02)

    mn, mx = [out['scalars'].min(), out['scalars'].max()]
    clim = (mn, mx * 1.8)

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(
        out,
        cmap='gist_earth_r',
        # background='white',
        show_scalar_bar=False,
        lighting=True,
        clim=clim,
        show_edges=False,
    )

    pl.camera = camera

    screenshots.append(pl.show(
        return_img=True,
    ))

imageio.mimsave('terrains.gif', screenshots)

import pyvista as pv
import imageio
from tqdm.auto import tqdm


NUM_ITERS = 60 # number of video frames

# initialise perlin noise for the first time
freq = (1, 1, 1)
noise = pv.perlin_noise(1, freq, (0, 0, 0))

# sample grid with perin noise and threshold it
grid = pv.sample_function(noise, [0, 3.0, -0, 1.0, 0, 1.0], dim=(120, 40, 40))
out = grid.threshold(0.02)

# get colormap limits for plotting
mn, mx = [out['scalars'].min(), out['scalars'].max()]
clim = (mn, mx * 1.8)

# initialise plotter
pl = pv.Plotter(off_screen=True)
pl.add_mesh(out)

# save camera position to avoid images jumping in the video
camera = pl.camera

# save the screenshots sequence
screenshots = []
for i in tqdm(range(1, NUM_ITERS + 1)):

    noise = pv.perlin_noise(1, freq, (i/NUM_ITERS * 2, 0, 0))
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

    # re-use old camera
    pl.camera = camera

    screenshots.append(pl.show(
        return_img=True,
    ))

imageio.mimsave('terrains.gif', screenshots)

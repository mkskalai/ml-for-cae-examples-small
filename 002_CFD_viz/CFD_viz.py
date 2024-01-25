import numpy as np
import imageio
from tqdm.auto import tqdm

import pyvista as pv
from pyvista import examples

block = examples.download_openfoam_tubes()

# first, get the first block representing the air within the tube.
air = block[0]
inlet = block[1][2]

num_slices = 30

z_min = min(air.points[:,-1])
z_max = max(air.points[:,-1])
step = (z_max - z_min) / num_slices

screenshots = []
for i in tqdm(range(1, num_slices)):

    # generate a slice in the XY plane
    z_slice = air.slice('z', origin=[0, 0, z_min + i * step])

    pset = pv.PointSet(inlet.points[::5])
    lines = air.streamlines_from_source(
        pset,
        vectors='U',
        max_time=1.0,
    )

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(z_slice, scalars='U', lighting=False, scalar_bar_args={'title': 'Flow Velocity'}, rng=[0,212])
    pl.add_mesh(
        lines,
        render_lines_as_tubes=True,
        line_width=3,
        lighting=False,
        scalar_bar_args={'title': 'Flow Velocity'},
        scalars='U',
        rng=(0, 212),
    )
    pl.add_mesh(air, color='w', opacity=0.25)
    pl.enable_anti_aliasing()
    # pl.show()
    screenshots.append(pl.show(return_img=True))

imageio.mimsave('CFD_viz.gif', screenshots)

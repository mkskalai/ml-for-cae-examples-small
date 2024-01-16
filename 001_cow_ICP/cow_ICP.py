import pyvista as pv
from pyvista import examples
import imageio

source = examples.download_cow().triangulate()
source = source.rotate_x(90)

pl = pv.Plotter(off_screen=True)
pl.add_mesh(source)
pl.show()
camera = pl.camera_position

transformed = source.rotate_y(-30).translate([-1.75, -0.5, 1.5])
aligned = transformed.align(source)

translation = aligned.points - transformed.points

screenshots = []
num_iters = 30
for i in range(num_iters):
    pl = pv.Plotter(off_screen=True)
    _ = pl.add_mesh(
        source, style='wireframe', opacity=0.5, line_width=1, color="red",
    )
    transformed.points += translation / num_iters
    _ = pl.add_mesh(transformed)
    
    pl.show_axes_all()
    pl.camera_position = camera
    screenshots.append(pl.show(return_img=True))

imageio.mimsave('cow_ICP.gif', screenshots)

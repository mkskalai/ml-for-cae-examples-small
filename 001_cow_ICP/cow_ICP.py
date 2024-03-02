import pyvista as pv
from pyvista import examples
import imageio

# download Cow mesh and position it more naturally for iso view
source = examples.download_cow().triangulate()
source = source.rotate_x(90)

# save the initial camera position so that images do not jump in the gif
pl = pv.Plotter(off_screen=True)
pl.add_mesh(source)
pl.show()
camera = pl.camera_position

# transform the source mesh and run PyVista's ICP to align it back
transformed = source.rotate_y(-30).translate([-1.75, -0.5, 1.5])
aligned = transformed.align(source)

# Get the translations of each node
translation = aligned.points - transformed.points

# create 30 frames where transformed mesh is gradally 
# moved back to it's original position
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

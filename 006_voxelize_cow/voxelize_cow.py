import numpy as np
import pyvista as pv
from tqdm.auto import tqdm
import imageio


# load and position mesh
mesh = pv.examples.download_cow()
mesh.rotate_x(90, inplace=True)
mesh.rotate_z(90, inplace=True)

images = []
for i in tqdm(np.geomspace(10, 250, num=30)):
    # cell size decreases in a geometric progression
    voxels = pv.voxelize(mesh, density=mesh.length / i, check_surface=False)
    # get the largest mesh to remove lose artifacts
    voxels_surface = voxels.extract_surface().extract_largest()
    # avoid plotting scalar field
    voxels_surface.clear_data()
    images.append(voxels_surface.plot(off_screen=True, return_img=True))
    
imageio.mimsave("cow_voxelized.gif", images, loop=0)

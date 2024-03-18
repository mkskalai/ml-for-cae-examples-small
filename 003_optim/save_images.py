import os
from pathlib import Path
import numpy as np
import pyvista as pv
import imageio
from tqdm.auto import tqdm
from PIL import Image
import argparse
from multiprocessing import Pool
from functools import partial

pv.global_theme.point_size = 10



def filter_pts(pts, lim):
    pts = pts[np.logical_and(
        (np.logical_and(pts[:,0] < lim, pts[:,0] > -lim)), 
        (np.logical_and(pts[:,1] < lim, pts[:,1] > -lim))
    )]
    return pts

        
def plot_population_generic(savedir, mesh, lim, overwrite, text, population, id_):
    file = savedir / f"{id_}.png"
    if os.path.isfile(file) and not overwrite:
        return
    pts = pv.PolyData(filter_pts(population, lim))
    p = pv.Plotter(off_screen=True)
    p.add_mesh(mesh)
    p.add_mesh(pts, color="red")
    p.add_text(f"iter: {id_}", position="upper_right", font_size=14)
    p.add_text(text, position="upper_edge", font_size=18)
    img = Image.fromarray(p.show(return_img=True))
    img.save(file)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-r", 
        "--root", 
        action="store", 
        dest='root', 
        help='folder containing the surface mesh and optimization populations', 
        required=True,
        type=str,
    )
    parser.add_argument(
        "-np", 
        "--num_processes", 
        action="store", 
        dest='num_processes', 
        help='number of parallel processes', 
        default=1, 
        required=True,
        type=int,
    )
    parser.add_argument(
        "-t", 
        "--text", 
        action="store", 
        dest='text', 
        help='text to put in the middle of the image', 
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o", 
        "--overwrite", 
        action="store_true", 
        dest='overwrite', 
        help='folder containing the surface mesh and optimization populations', 
        required=False,
    )
    
    args = parser.parse_args()
    
    root = Path(args.root)
    populations_dir = root / "populations"
    paths = []
    for path in populations_dir.glob("*.npy"):
        paths.append(path)
    paths.sort()
    
    populations = []
    ids = []
    for path in paths:
        populations.append(np.load(path))
        ids.append(path.stem)
    
    mesh = pv.read(root / "surf.stl")
    lim = max(mesh.points[:,0])
    savedir = root / "images"
    savedir.mkdir(exist_ok=True)
    plot_population = partial(
        plot_population_generic, 
        savedir, 
        mesh, 
        lim, 
        args.overwrite, 
        args.text
    )
    inputs = zip(populations, ids)
    with Pool(args.num_processes) as pool:
        _ = pool.starmap(plot_population, tqdm(inputs, total=len(populations)))
                
    images = []
    for id_ in ids:
        images.append(Image.open(savedir / f"{id_}.png"))
    imageio.mimsave(root / "optimization.gif", images)
    
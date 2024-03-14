from vtk import read_vtk
from get_thickness_white2pial import get_thickness_white2pial


# file path --------------------------------------- 

lh_white_path = "./surf/lh.CortexODE.white.vtk"
lh_pial_path = "./surf/lh.CortexODE.pial.vtk"
rh_white_path = "./surf/rh.CortexODE.white.vtk"
rh_pial_path = "./surf/rh.CortexODE.pial.vtk"


# load file ---------------------------------------

lh_white = read_vtk(lh_white_path)
lh_pial = read_vtk(lh_pial_path)
rh_white = read_vtk(rh_white_path)
rh_pial = read_vtk(rh_pial_path)

# GET DISTANCE -----------------------------------

new_lh_white, new_rh_white = get_thickness_white2pial(lh_white,lh_pial,rh_white,rh_pial,combine = False)



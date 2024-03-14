from numba import jit
import numpy as np
from vtk import read_vtk,write_vtk,to_polydata,write_vertices,remove_field
from tqdm import tqdm
import trimesh
from queue import PriorityQueue

def get_thickness_white2pial(lh_white,lh_pial,rh_white,rh_pial,combine = False):


    # split the data to vertices and faces ------------

    lh_white_vertices = lh_white['vertices']
    lh_white_faces = lh_white['faces'][:,1:]

    rh_white_vertices = rh_white['vertices']
    rh_white_faces = rh_white['faces'][:,1:]

    lh_pial_vertices = lh_pial['vertices']
    lh_pial_faces = lh_pial['faces'][:,1:]

    rh_pial_vertices = rh_pial['vertices']
    rh_pial_faces = rh_pial['faces'][:,1:]


    # make mesh data using trimesh --------------------

    lh_white_mesh = trimesh.Trimesh(vertices=lh_white_vertices, faces=lh_white_faces)
    rh_white_mesh = trimesh.Trimesh(vertices=rh_white_vertices, faces=rh_white_faces)
    lh_pial_mesh = trimesh.Trimesh(vertices=lh_pial_vertices, faces=lh_pial_faces)
    rh_pial_mesh = trimesh.Trimesh(vertices=rh_pial_vertices, faces=rh_pial_faces)


    # Get the vertex normals --------------------------

    lh_white_vertex_normals = lh_white_mesh.vertex_normals
    rh_white_vertex_normals = rh_white_mesh.vertex_normals


    # Make pial surface mesh intersector using trimesh -

    lh_pial_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(lh_pial_mesh)
    rh_pial_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(rh_pial_mesh)

        
    if combine == False : 
        print('----lh thickness calculation start----')
        lh_new_thickness = get_distance(
                                 lh_white_vertices,
                                 lh_white_vertex_normals,
                                 lh_pial_intersector)
        lh_white['new_thickness'] = np.array(lh_new_thickness)
        write_vtk(lh_white,'lh_white_not_combined.vtk')
        
        print('----rh thickness calculation start----')
        rh_new_thickness = get_distance(
                                rh_white_vertices,
                                rh_white_vertex_normals,
                                rh_pial_intersector)


        rh_white['new_thickness'] = np.array(rh_new_thickness,dtype=np.float64)
        write_vtk(rh_white,'rh_white_not_combined.vtk')
        
        return lh_white,rh_white
    
    else : 
        # concatenate lh & rh brain -----------------------

        combined_white_vertices =  np.concatenate((lh_white_vertices, rh_white_vertices), axis=0)
        combined_white_vertices_norm =  np.concatenate((lh_white_vertex_normals, rh_white_vertex_normals), axis=0)

        combined_pial_mesh = trimesh.util.concatenate(lh_pial_mesh, rh_pial_mesh)
        combined_pial_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(combined_pial_mesh)
        
        print('----rh & lh combine thickness calculation start----')
        new_thickness = get_distance(
                                 combined_white_vertices,
                                 combined_white_vertices_norm,
                                 combined_pial_intersector)

        len_white_vetices = len(lh_white_vertices)
        lh_white['new_thickness'] = np.array(new_thickness[:len_white_vetices],dtype=np.float64)
        rh_white['new_thickness'] = np.array(new_thickness[len_white_vetices:],dtype=np.float64)
   
        write_vtk(lh_white,'lh_white_combined.vtk')
        write_vtk(rh_white,'rh_white_combined.vtk')
        
        return lh_white,rh_white
    

    

# make a function to get distance from white matter to pial surfaces using trimesh. 

def get_distance(white_vertices,white_vtx_normal,pial_mesh_intersector):
    
    not_intersect_num = 0 
    new_thickness = []
    
    for i,(ray_origins,ray_directions) in tqdm(enumerate(zip(white_vertices,white_vtx_normal)),total=len(white_vertices)) : 
        
        ray_origins = ray_origins.reshape(1,3)
        ray_directions = ray_directions.reshape(1,3)
        locations = pial_mesh_intersector.intersects_location(ray_origins, ray_directions)
        
        intersect_positions = locations[0]
        num_faces_intersect = len(locations[1])
        intersect_faces_idx = locations[2]
        pq = PriorityQueue()
        
        if num_faces_intersect == 0 : 
            not_intersect_num += 1
            new_thickness.append(np.nan)
            
        else : 
            for j,intersect_pos in enumerate(intersect_positions) : 
                
                dist = np.sqrt(np.sum((ray_origins - intersect_pos)**2))
                pq.put(dist,{'distance':dist})
            small_one = pq.get()
            new_thickness.append(np.round(small_one.astype(float),8))
            
                
    print(f"Vertex num : {len(white_vertices)}")
    print(f"Not intersect num : {not_intersect_num}")
    return new_thickness  # list form

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:17:52 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import copy
import os

import numpy as np
import pyvista

abspath = os.path.abspath(os.path.dirname(__file__))


def read_vtk(in_file):
    """
    Read .vtk POLYDATA file
    
    in_file: string,  the filename
    Out: dictionary, 'vertices', 'faces', 'curv', 'sulc', ...
    """

    polydata = pyvista.read(in_file)
 
    n_faces = polydata.n_faces
    vertices = np.array(polydata.points)  # get vertices coordinate
    
    # only for triangles polygons data
    faces = np.array(polydata.GetPolys().GetData())  # get faces connectivity
    assert len(faces)/4 == n_faces, "faces number is not consistent!"
    faces = np.reshape(faces, (n_faces, 4))
    
    data = {'vertices': vertices,
            'faces': faces
            }
    
    point_data = polydata.point_data
    for key, value in point_data.items():
        if value.dtype == 'uint32':
            data[key] = np.array(value).astype(np.int64)
        elif value.dtype == 'uint8':
            data[key] = np.array(value).astype(np.int32)
        else:
            data[key] = np.array(value)

    return data
    

def write_vtk(in_dic, file, vertices='vertices'):
    """
    Write .vtk POLYDATA file
    
    in_dic: dictionary, vtk data
    file: string, output file name
    """
    surf = to_polydata(in_dic, vertices=vertices)
    surf.save(file, binary=False)


def to_polydata(in_dic, vertices='vertices'):
    assert vertices in in_dic, "output vtk data does not have %s!" % vertices
    assert 'faces' in in_dic, "output vtk data does not have faces!"

    data = copy.deepcopy(in_dic)

    v = data[vertices].astype(np.float32)
    f = data['faces']
    surf = pyvista.PolyData(v, f)

    if vertices == 'vertices':
        del data[vertices]
    del data['faces']
    for key, value in data.items():
        if isinstance(value.dtype, np.floating):
            value = value.astype(np.float32)
        surf.point_data[key] = np.nan_to_num(value)
    return surf
    
    
def write_vertices(in_ver, file):
    """
    Write .vtk POLYDATA file
    
    in_dic: dictionary, vtk data
    file: string, output file name
    """
    
    with open(file,'a') as f:
        f.write("# vtk DataFile Version 4.2 \n")
        f.write("vtk output \n")
        f.write("ASCII \n")
        f.write("DATASET POLYDATA \n")
        f.write("POINTS " + str(len(in_ver)) + " float \n")
        np.savetxt(f, in_ver)


def remove_field(data, *fields):
    """
    remove the field attribute in data
    
    fileds: list, strings to remove
    data: dic, vtk dictionary
    """
    for field in fields:
        if field in data.keys():
            del data[field]
    
    return data

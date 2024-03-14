import vtk
def write_lines_to_vtk(points, filename):
    """
    Write lines to a VTK file using pyvista.
    
    Parameters:
    - points: List of point pairs. Each point pair is a tuple of two 3D points.
    - filename: Name of the output VTK file.
    
    # Example usage
        points = [
            ((0, 0, 0), (1, 1, 1)),
            ((2, 2, 2), (3, 3, 3)),
            ((4, 4, 4), (5, 5, 5)),
        ]
    """
    
    # Create a new VTK points structure
    points_vtk = vtk.vtkPoints()
    
    # Create a new VTK cell array to store the lines
    lines = vtk.vtkCellArray()
    
    index = 0
    for point1, point2 in points:
        points_vtk.InsertNextPoint(point1)
        points_vtk.InsertNextPoint(point2)
        
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, index)
        line.GetPointIds().SetId(1, index + 1)
        
        lines.InsertNextCell(line)
        index += 2

    # Create a new VTK polydata structure
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points_vtk)
    polydata.SetLines(lines)

    # Write the data to file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()



def write_points_to_vtk(points_list, filename="output.vtk",color=(255,0,0)):
    """
    Render colored points using VTK and save to a VTK file.
    
    Args:
    - points_list: List of points where each point is a tuple (x, y, z).
    - filename: Name of the VTK file to save.
    - color: RGB color tuple (R, G, B) to use for all points.
    """
    
    # 1. Create points
    points = vtk.vtkPoints()
    for pt in points_list:
        points.InsertNextPoint(pt)

    # 2. Assign a single color for all points
    num_points = len(points_list)
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    for _ in range(num_points):
        colors.InsertNextTuple(color)

    # 3. Set up PolyData structure
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetScalars(colors)

    # Save to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

    # # If you wish to also visualize the points (optional):
    # # 4. Map data to graphics primitives
    # mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputData(polydata)

    # # 5. Create an actor
    # actor = vtk.vtkActor()
    # actor.SetMapper(mapper)
    # actor.GetProperty().SetPointSize(10)  # Adjust point size if needed

    # # 6 & 7. Set up a renderer and add the actor
    # renderer = vtk.vtkRenderer()
    # renderer.AddActor(actor)

    # # 8. Set up a render window
    # renderWindow = vtk.vtkRenderWindow()
    # renderWindow.AddRenderer(renderer)

    # # 9. Interact with the scene
    # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    # renderWindowInteractor.SetRenderWindow(renderWindow)
    # renderWindow.Render()
    # renderWindowInteractor.Start()

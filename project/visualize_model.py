import SimpleITK as sitk
import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np

# Load the CT scan
file_path = 'path/to/your/ct_scan.mha'
ct_image = sitk.ReadImage(file_path)

# Preprocess the image (optional)
ct_image = sitk.Median(ct_image)

# Segment the image
ct_array = sitk.GetArrayFromImage(ct_image)
lower_threshold = 200
upper_threshold = np.max(ct_array)
binary_mask = np.logical_and(ct_array >= lower_threshold, ct_array <= upper_threshold)
binary_image = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
binary_image.CopyInformation(ct_image)

def generate_surface(binary_image):
    # Convert SimpleITK image to VTK image
    spacing = binary_image.GetSpacing()
    origin = binary_image.GetOrigin()
    
    binary_array = sitk.GetArrayFromImage(binary_image)
    vtk_data_array = vtk_np.numpy_to_vtk(num_array=binary_array.ravel(order='F'), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(binary_array.shape[::-1])
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.GetPointData().SetScalars(vtk_data_array)
    
    # Apply Marching Cubes
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_image)
    mc.SetValue(0, 1)
    mc.Update()
    
    # Smooth the surface
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(mc.GetOutputPort())
    smoother.SetNumberOfIterations(15)
    smoother.SetRelaxationFactor(0.1)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.Update()
    
    return smoother.GetOutput()

def visualize_model(poly_data):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)
    
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    render_window.Render()
    interactor.Initialize()
    interactor.Start()

def save_model(poly_data, filename):
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(filename)
    stl_writer.SetInputData(poly_data)
    stl_writer.Write()

# Generate the surface
surface_model = generate_surface(binary_image)

# Visualize the model
visualize_model(surface_model)

# Save the model
output_filename = 'ct_model.stl'
save_model(surface_model, output_filename)

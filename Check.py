import numpy as np
import pyvista as pv
from pyvista import examples
import pydicom
import pandas as pd
import os
import pydicom as dicom
from pydicom import dcmread
import skimage
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects , binary_closing, disk
import matplotlib.pyplot as plt
from stl import mesh

folderpath = "/Users/jordanedmond/Desktop/BMEN480FinalProject/Images/anklebonepractice_files" #change this file depending on set of dicom images

def make_dicom_list(folderpath):  #makes list of all dicom files in folder
    dicom_list = [file for file in os.listdir(folderpath) if file.endswith('.dcm')]
    dicom_list = [dicom.dcmread(os.path.join(folderpath, file)) for file in dicom_list]
    dicom_list = sorted(dicom_list, key=lambda x: int(x.InstanceNumber))
    return dicom_list

#def check_dimensions(dicomfile):
 #   ds = dicom.dcmread(dicomfile)
  ## columns = ds.Columns
   # print(rows)
   # print(columns)

def create_array(dicom_index): #Creates array
    slices = np.zeros([512, 512, len(dicom_index)])
    for i, slice in enumerate(dicom_index):  # iterates dicom list
        slices[:, :, i] = slice.pixel_array #[all rows, all columns, depth based on index]
    return slices

def segmentationprocess(image):
    threshold = threshold_otsu(image)
    binaryimage = image > threshold
    clearedbinary = clear_border(binaryimage)  #
    filteredbinary = remove_small_objects(clearedbinary, min_size=200)  # removes objects smaller than a certain size
    smoothedbinary = binary_closing(filteredbinary, footprint=disk(5))
    return smoothedbinary

def comparison(originalimage,segmentedimage):
    for i in range(originalimage.shape[2]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(originalimage[:, :, i], cmap='gist_gray')
        ax[0].set_title(f"Original Slice {i + 1}/{originalimage.shape[2]}")
        ax[1].imshow(segmentedimage[:, :, i], cmap='gist_gray')
        ax[1].set_title(f"Segmented Slice {i + 1}/{segmentedimage.shape[2]}")
        plt.show()

def segmentation_over_all(segmentedslices): #Runs segmentationprocess function on all the slides
    segmented_slices = np.zeros_like(segmentedslices)
    for i in range(segmentedslices.shape[2]):
        segmented_slices[:,:,i] = segmentationprocess(segmentedslices[:,:,i])
    return segmented_slices

def make_points(dicom_data,subset=.02):
    indices = np.argwhere(dicom_data > 0)  # Extract 3D coordinates of nonzero points
    n_points = indices.shape[0]
    ids = np.random.randint(low=0, high=n_points - 1, size=int(n_points * subset))
    return indices[ids]

def create_point_cloud(array): #Creates point cloud and mesh
    points = make_points(array).astype(np.float32)
    point_cloud = pv.PolyData(points)
    np.allclose(points, point_cloud.points)
    point_cloud.plot(eye_dome_lighting=True)
    return point_cloud

def mesh_to_stl(mesh):
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, "anklebone.stl")
    mesh.save(file_path)
    print(f"File saved to {file_path}")

dicom_files = make_dicom_list(folderpath) #runs folder path and puts files into list
slices_array = create_array(dicom_files) #creates an array of original image slices
segmented_image_array = segmentation_over_all(slices_array)  #creates array of segmented slices
#comparison(slices_array, segmented_image_array) #visualizes original vs segmented images in plot
finished_point_cloud = create_point_cloud(segmented_image_array)
mesh_to_stl(finished_point_cloud)





#Creates point cloud and mesh
#points = make_points(segmented_image_array).astype(np.float32) #makes points from segemented image array into float32 from int32
#point_cloud = pv.PolyData(points) #Mesh is create from points with PyVista function
#np.allclose(points, point_cloud.points)
#point_cloud.plot(eye_dome_lighting = True) #eye dome lighting is a common shading practice that improves depth perception


#checking dimensions and verifying
print(slices_array[1])
print(slices_array[1].shape)
print(slices_array.shape)
print(segmented_image_array[1].shape)
print(segmented_image_array.shape)

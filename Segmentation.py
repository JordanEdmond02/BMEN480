import dash_bootstrap_components
import pandas
import numpy as np
import dash
from dash import Dash, html, dcc, callback, Output, Input
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import plotly.express as px
import skimage
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects , binary_closing, disk
import pandas as pd
import pydicom as dicom
from pydicom import dcmread
import matplotlib.pyplot as plt
import base64
import pybase64
import os
import dicom2jpg
import io
import vtk
from vtkmodules.vtkCommonDataModel import vtkImageData

#Segmentation
dicom_list = os.listdir("/Users/jordanedmond/Desktop/BMEN480FinalProject/Images/anklebonepractice_files")
sorteddicomlist = np.zeros([512,512,len(dicom_list)])  #512x512 is usually size of each slice
dicom_list = [dicom.dcmread(os.path.join("/Users/jordanedmond/Desktop/BMEN480FinalProject/Images/anklebonepractice_files",file)) for file in dicom_list]
dicom_list = sorted(dicom_list, key = lambda x: int(x.InstanceNumber))
for i, slice in enumerate(dicom_list):
    sorteddicomlist[:,:,i] = slice.pixel_array  #[:,:,i] says for each slice in the list

for i in range(sorteddicomlist.shape[2]):#sorteddicomlist is a dataframe and it is in 2 rows and shape is the amount of dimensions(2D is 1, 3D is 2)

    #Getting Binary Images
    image = sorteddicomlist[:,:,i]
    threshold = threshold_otsu(image)
    binaryimage = image > threshold
    clearedbinary = clear_border(binaryimage) #cleans image
    filteredbinary = remove_small_objects(clearedbinary, min_size =200) #filters out extra parts of image
    smoothedbinary = binary_closing(filteredbinary, footprint=disk(5)) #smooths image

    fig, ax = plt.subplots(1,2, figsize = (10,5))
    ax[0].imshow(sorteddicomlist[:,:,i], cmap='gist_gray')
    ax[0].set_title(f"Original Slice {i + 1}/{sorteddicomlist.shape[2]}")
    ax[1].imshow(smoothedbinary, cmap = 'gist_gray')
    ax[1].set_title(f"Segmented Slice {i+1}/{sorteddicomlist.shape[2]}")
    plt.show()





READ ME


# Segmentation and 3D Surface Reconstruction of Bones from CT Scans

Download Python 3.11 to run Open3D

## Problem Statement:
Reliance on 2D image slices for organ visualization, particularly in the context of bone analysis through CT scans, significantly limits the accuracy of anatomical assessments and understanding of anatomical structures. 

## Need Statement:
Radiologists need to enhance radiology assessments of internal bone structure to increase the efficiency of treatment planning. 

## Necessary Packages to install:
-dash 
  -import dash_bootstrap_components as dbc
  -from dash import Dash, html, dcc, callback, Output, Input
  
-pandas

-numpy

-pydicom
  -from pydicom import dcmread
  
-io

-os

-skimage
  -from skimage.filters import threshold_otsu
  
-vtk

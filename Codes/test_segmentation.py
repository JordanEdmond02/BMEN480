import numpy as np
import os
import shutil
import pytest
from final_code import segmentationprocess, segmentation_over_all, create_array, clean_old_files

def test_segmentationprocess_output():
    image = np.random.rand(100, 100) #dummy 2D image is fet
    result = segmentationprocess(image)
    assert isinstance(result, np.ndarray) #checks that result is numpy array
    assert result.shape == image.shape #checks that shape matches the output
    assert result.dtype == bool #checks that datatype is boolean, because mask is binary

def test_segmentation_over_all_output():
    volume = np.random.rand(64, 64, 20) #creates a dummy 3D volume
    result = segmentation_over_all(volume)
    assert isinstance(result, np.ndarray) #checks that numpy array is returned
    assert result.shape == volume.shape #checks that shape is the same as the input volume
    assert result.dtype == bool #checks that each voxel is binary

def test_create_array_output(): #checks that the array is created correctly
    class MockDICOM:
        def __init__(self, array):
            self.pixel_array = array
    slices = [MockDICOM(np.random.randint(0, 255, (64, 64))) for _ in range(5)]
    result = create_array(slices)
    assert result.shape == (64, 64, 5)
    assert result.dtype == np.float32

def test_clean_old_files(tmp_path): #checks function that empties directory
    temp_dir = tmp_path / "temp"  #creates a temp directory
    temp_dir.mkdir()
    for i in range(3):
        (temp_dir / f"file{i}.txt").write_text("test")
    clean_old_files(str(temp_dir))
    assert len(os.listdir(temp_dir)) == 0


import cv2
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd 
import geopandas as gpd 

import rasterio       # For reading .tif files
import matplotlib.pyplot as plt  # For plotting
import numpy as np
import pdb 
import xarray as xr

import rioxarray as xrio

from affine import Affine
from rasterio.transform import from_origin
from rasterio.features import geometry_mask


def aligned_terrain(terrain_f, terrain_srtm):
    
    im2_gray = terrain_f.values.astype(np.float32)
    im1_gray = terrain_srtm.values.astype(np.float32)
    
    # Create a mask to include the area of interest
    mask = np.zeros(im2_gray.shape, dtype=np.uint8)
    idxdata = np.where(im2_gray>-50)
    mask[idxdata] = 1  # Example mask region
    #shrink to avoid border effect
    #mask[0,:]=0
    #mask[:,0]=0
    #mask[-1,:]=0
    #mask[:,-1]=0
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((51,51)))
    
    # Find size of image1
    sz = im1_gray.shape
     
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
     
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
     
    # Specify the number of iterations.
    number_of_iterations = 10000;
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
     
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
     
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray,im2_gray,warp_matrix, warp_mode, criteria, mask)
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        mask_aligned = cv2.warpAffine(mask, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,borderValue=0);
    
    # Show final results
    terrain_f_aligned =im2_aligned
    
    #update terrain_f
    terrain_f_old = terrain_f.copy()
    terrain_f.values = np.where(mask_aligned==0,-9999,terrain_f_aligned)    

    return  terrain_f, warp_matrix#, translation



def create_mask_from_ba( da, ba):

    latitude = da['y'].values  # expect reverse order in lat
    longitude = da['x'].values

    #define the transformation
    min_lon = longitude.min()
    max_lat = latitude.max()
    reslon  = np.diff(longitude).mean()
    reslat = np.diff(latitude).mean()
    height, width = latitude.shape[0], longitude.shape[0]

    transform = Affine.translation(longitude.min(), latitude.max()) * Affine.scale(reslon, reslat)

    # Convert geometries to the rasterio format
    shapes = [geom for geom in ba.to_crs(da.rio.crs).geometry]

    maskBa = geometry_mask(shapes, transform=transform, invert=True, 
                         out_shape=(height, width),all_touched=True).astype(np.uint8)


    return maskBa

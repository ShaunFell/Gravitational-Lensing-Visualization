import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as ptch
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
import astropy.constants as c
import astropy.units as u
import random
import numpy as np
import math
import os
plt.style.use('grayscale')

radtoarcsec = 206265
arcsectorad = 1/radtoarcsec



#Open the image file as a matrix using matplotlibs imread
image_file = "./src/eso1738e.jpg" #relative path to image file
image_data_unshapped = img.imread(image_file) #import image file as a matrix. Each component of the NxM matrix is a length 3 array containing the RGB values of the pixel.
image_min_side_inxlength = np.min(image_data_unshapped.shape[:2]) #find the minimum length of the image matrix. I.E. for an NxM matrix, which is smaller the 'N' or the 'M'?
image_data = image_data_unshapped[:image_min_side_inxlength,:image_min_side_inxlength] #crop the matrix using the minimum side length so that it is square. I.E. if 'N' is smaller, then reshape to an NxN matrix.
source_data = np.empty(image_data.shape)

#setup the System
D_L = 3000 # Distance to black hole in parsecs
D_S = 5000 # Distance to source object in parsecs. This assumes the stars in the image are distributed planarly and they are much further away from the hole than they are from each other. This assumption makes the math much easier
D_LS = np.abs(D_S-D_L) #parsecs. distance from source to hole
G = c.G.value #gravitational constant
M = 10**13 #solar masses. Mass of lensing hole
C = c.c.value #m/s
R_E = np.sqrt((4*G*M*D_LS)/((C**2)*D_L*D_S))*D_L*radtoarcsec #Einstein Radius in arcseconds


# define and calculate image parameters
image_Ang_Size = 2*R_E # Image angular size in arcseconds
image_Mat_Len = image_min_side_inxlength #rename matrix size variable
image_pixel_Center = np.array([image_Mat_Len/2, image_Mat_Len/2])
image_Length = 2*np.tan(image_Ang_Size*arcsectorad)*D_S # Image actual size in parsecs. Should be about 0.00714 pc
pixel_Size = image_Length/image_min_side_inxlength #side length of a pixel in parsecs


def invlensed_pixel(pixel_list):
    """pixel list should be relative to top left"""
    pix_angle_from_center = math.atan(math.sqrt((pixel_list[0]-image_pixel_Center[0])**2 + (pixel_list[1]-image_pixel_Center[1])**2)*pixel_Size/D_S)*radtoarcsec
    sourceangl = pix_angle_from_center - R_E**2/pix_angle_from_center #arcseconds
    source_pix = [
                    (pixel_list[0]-image_pixel_Center[0])*sourceangl/pix_angle_from_center+image_pixel_Center[0],
                    (pixel_list[1]-image_pixel_Center[1])*sourceangl/pix_angle_from_center+image_pixel_Center[1]
                ]
    
    return [round(source_pix[0]), round(source_pix[1])]


def lensed_image(imagedata):
    ret = np.empty(imagedata.shape,dtype=np.uint8)
    for i in range(image_Mat_Len):
        for j in range(image_Mat_Len):
            if (i == image_pixel_Center[0] and j == image_pixel_Center[1]):
                continue
            inverse_lensed_pix = invlensed_pixel([i,j])
            if inverse_lensed_pix[0]<0 or inverse_lensed_pix[0]>image_Mat_Len-1 or inverse_lensed_pix[1]<0 or inverse_lensed_pix[1]>image_Mat_Len-1:
                continue
            if inverse_lensed_pix[0]>-1 and inverse_lensed_pix[1]>-1:
                ret[i,j] = imagedata[inverse_lensed_pix[0], inverse_lensed_pix[1]]
    return ret


lensedimage = lensed_image(image_data)

fig = plt.figure(figsize=(20,20))
fig.add_subplot(1,2,1)
plt.imshow(image_data)
fig.add_subplot(1,2,2)
plt.imshow(lensedimage)

plt.show()

#pip install image_slicer
import image_slicer
import PIL
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import random
import os

# Path to white damage folder
white_path = '/home/link92/damage_insertion/venv/damages/white_noise'
# Path to black damage folder
black_path = '/home/link92/damage_insertion/venv/damages/black_noise'


def create_crops(path, mode="white"):
	"""
	path: path to damages folder
	mode: str indicating whether the damage should be treated as "white" or "black"
	"""
	
	os.chdir(path)
	if not os.path.exists('output'):
        	os.mkdir('output')
        
        # Take the path of .jpg damages
       	starting_images = [f for f in glob(f'{path}/*dmg[0-9].jpg')]
       	
       	# For each damage path
    	for damage_path in starting_images:
    		
    		# Open it as image and take its file name
        	curr_dmg = cv2.imread(damage_path, cv2.IMREAD_GRAYSCALE)
        	curr_name = damage_path.split('/')[-1].replace('.jpg', '')
		
		# take its x and y dimensions
       	 	m, n = curr_dmg.shape
       	 	
       	 	# Create an alpha channel
       	 	alpha = np.full((m, n), 255, dtype='uint8')
       	 	
        	if mode == "white":
        		# Impose the pixels further from white color to be black
            		curr_dmg[curr_dmg < 125] = 0            		
            		# Impose alpha channel (make it invisible) where the image is black
            		alpha[np.where(curr_dmg == 0)] = 0
        	else:
        		# Impose the pixels further from black to be white
            		curr_dmg[curr_dmg >= 125] = 255
            		# Impose alpha channel (make it invisible) where the image is white
            		alpha[np.where(curr_dmg == 255)] = 0
		
		# Merge all the channels of the image so to make it RGBA
        	curr_dmg_rgba = cv2.merge([curr_dmg, curr_dmg, curr_dmg, alpha], 4)
        	
        	# Create a temporary folder where to put the crops
        	if not os.path.exists('temp'):
            		os.mkdir('temp')
            		
            	# Write the created RGBA image as .png
        	cv2.imwrite(f'temp/alpha_{curr_name}.png', curr_dmg_rgba)
        	
		# Reopen it and slice it in 32 (actually 36) equal parts, store in a temp folder
        	image_slicer.slice(f'temp/alpha_{curr_name}.png', 32)
		
		# Take each created crop, copy it in the output folder, and delete the temp folder
        	for i in glob(f'{path}/temp/*_[0-9][0-9].png'):
            		os.system(f'cp {i} {path}/output')
        	os.system(f'rm -rf {path}/temp')

#create_crops(white_path, "white")
#create_crops(black_path, "black")

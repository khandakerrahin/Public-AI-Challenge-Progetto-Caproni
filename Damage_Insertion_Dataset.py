import PIL
import cv2
from PIL import Image
import numpy as np
from glob import glob
import random
import os


def superimpose(background, choices, image_name):
	"""
	background: cv2.image (the image to which to impose the damage)
	choices: path of the chosen crop damages
	image_name: str, determines how the file should be stored
	"""
	
	# Save hor and ver pixels of the background image
	m, n, _ = np.array(background).shape
	
	# Create a black background for the mask
    	black_screen = np.zeros((m, n), 'uint8')
    	black_screen = Image.fromarray(black_screen).convert('RGBA')

	# Apply damage crops both to background (original image) and black_screen (mask image)
    	while choices:
    		# Determine a random angle to rotate the damage crop with
        	angle = random.randint(-179, 180)
        	# Determine two random integer to resize the damage crop
        	resize_x = random.randint(50, 255)
        	resize_y = random.randint(50, 255)
		
		# Take a damage crop from the previously chosen list of crop damages
        	curr = choices.pop()
        	# Save the name of the damage crop
        	curr_name = curr.split('/')[-1]
        	# Determine the mode according to the starting name of the damage crop
        	mode = curr_name.startswith('white')
        	# Open the damage crop with PIL and convert it as an RGBA
        	curr = PIL.Image.open(curr).convert('RGBA')
        	# Rotate it
        	curr = rotate(curr, angle)
        	# Change its size
        	curr = change_size(curr, (resize_x, resize_y))
		# Change random coordinates for placing it
        	coordx, coordy = choose_coords(background)
        	# Apply it to the original image
        	background.paste(curr, (coordx, coordy), mask=curr)

        	# save the imposed damages as an array and impose it to black_screen (the mask)
        	arr = np.array(curr)

        	if mode:
            		arr[arr >= 125] = 255
        	else:
            		arr[:, :, :3][arr[:, :, :3] < 125] = 255

        	arr = Image.fromarray(arr).convert('RGBA')

        	black_screen.paste(arr, (coordx, coordy), mask=arr)
	# Store the modified image and the mask
	# => CHANGE PATHS HERE <=
	# Paths here are harcoded, you should change them accordingly, except for '{image_name}.png'
    	background.save(f'/home/link92/Artificial_Damage_Dataset/venv/results/images/{image_name}.png')
    	black_screen.save(f'/home/link92/Artificial_Damage_Dataset/venv/results/masks/{image_name}.png')


def choose_coords(image):
	"""
	image: PIL.Image from which to choose the coordinate
	
	Outputs random x, y coordinates according to the original size of the image
	"""
	x, y = image.size
    	hor = random.randint(0, x)
    	ver = random.randint(0, y)
    	return hor, ver


def rotate(image, angle):
	"""
	image: PIL.Image to rotate
	angle: numerical value indicating the angle to rotate the image by
	
	Outputs a rotated image by the given angle, fill the portion of the image created with rotation as alpha
	"""
	return image.rotate(angle=angle, fillcolor=(0, 0, 0, 0))


def change_size(image, dim):
	"""
	image: PIL.Image of which to change size
	dim: tuple indicating how much to resize the image
	"""
	return image.resize(dim)


def create_dataset(damage_crop_path, max_n_damages=5):
	"""
	damage_crop_path: A path indicating where are the crops of damages
	max_n_damages: integer [0, +inf] indicating how many crops (at maximum) should be imposed on the image
	"""
	# Take the path of all possible crops of damages
    	list_of_crops = [crop for crop in glob(f'{damage_crop_path}/*.png')]
    	# Counter to store the images in a "name_file_{counter}" format
    	counter = 0
    	# => CHANGE PATH HERE <=
    	# The path here is hardcoded, you shold change it with your starting dataset path
    	for original_image in glob('/home/link92/Artificial_Damage_Dataset/venv/input_images/*'):
    		# Open the image as Grayscale + Alpha channel
        	curr_original = PIL.Image.open(original_image).convert('LA')
        	# Resize it @512x512
        	curr_original = curr_original.resize((512, 512))
        	# Store the name of it
        	curr_name = original_image.split('/')[-1][:-4]
        	# Determines a random number of damages to apply on curr_original (according to max_n_damages)
        	n_damages = random.randint(0, max_n_damages)
        	# Sample n_damages from all the possible damage crops
        	damages = [i for i in random.sample(list_of_crops, n_damages)]
        	# Impose the damage over curr_original, create a mask and
        	# Store the results in two distinct folders
        	superimpose(curr_original, damages, f'{curr_name}_{counter}')
        	counter += 1


#damage_crop_path = '/home/link92/Artificial_Damage_Dataset/venv/damage_crops'
#create_dataset(damage_crop_path)

import os
from utils import DamageDataset


root_dir = '/home/link92/PycharmProjects/pythonProject/create_damage_dataset/input_images'
results = '/home/link92/PycharmProjects/pythonProject/create_damage_dataset/results'
damages_dir = '/media/link92/E/damage_crops'


if not os.path.exists(results):
    os.mkdir(results)
    os.mkdir(f'{results}/image')
    os.mkdir(f'{results}/mask')

img, mask = DamageDataset(root_dir, damages_dir)[0]
img.save('/home/link92/PycharmProjects/pythonProject/create_damage_dataset/results/img.png')
mask.save('/home/link92/PycharmProjects/pythonProject/create_damage_dataset/results/mask.png')

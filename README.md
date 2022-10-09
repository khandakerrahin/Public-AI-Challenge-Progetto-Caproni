# Description of the scripts

## damage_crop.py

Starting from two folders, it is going to create crops of images. According to the folder in which the starting images are placed in:
* black_noise: In the images here, the **black part** of the image is taken as damage, the rest is made invisible
* white_noise: In the images here, the **white part** of the image is taken as damage, the rest is made invisible

Required structure of the folders (to make it work as it is):

```
| damage_crop.py
| damages
    |--- black_noise
            |--- black_damage_1.jpg
            |--- black_damage_2.jpg
            |--- ...
    |--- white_noise
            |--- white_damage_1.jpg
            |--- white_damage_2.jpg
            |--- ...
```

from PIL import Image
import numpy as np
import albumentations as A

class_diz = {'background': [0, 0, 0],  # black
             'bands': [255, 0, 0],  # red
             'stains': [0, 255, 0],  # green
             'dots': [0, 0, 255],  # blue
             'scratches': [255, 255, 255]  # white
             }


def one_hot_encode(label, label_values):
    semantic_map = []
    for color in label_values:
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    _transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
    return A.Compose(_transform)


def from_segmentation_to_mask(seg_map):
    m, n = seg_map.shape
    new_arr = np.zeros((m, n, 3), dtype='uint8')
    for i in range(m):
        for j in range(n):
            if seg_map[i, j] == 0:
                new_arr[i, j, :] = 0  # useless, just to be sure
            elif seg_map[i, j] == 1:
                new_arr[i, j, 0] = 255
            elif seg_map[i, j] == 2:
                new_arr[i, j, 1] = 255
            elif seg_map[i, j] == 3:
                new_arr[i, j, 2] = 255
    return new_arr


def reverse_one_hot(image):
    return np.argmax(image, axis=-1)


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    _transform.append(A.Lambda(image=to_tensor, mask=to_tensor))

    return A.Compose(_transform)


def mask_to_consider(lst):
    rgb_values = []

    for cls in lst:
        if cls in class_diz:
            rgb_values.append(class_diz[cls])
    return np.array(rgb_values)


# 'background': [0, 0, 0],  # black
# 'bands': [255, 0, 0],  # red
# 'stains': [0, 255, 0],  # green
# 'dots': [0, 0, 255],  # blue
# 'scratches': [255, 255, 255]  # white

def calculate_damage_score(damage_mask):
    # damage_mask = Image.fromarray(damage_mask)
    damage_mask_data = damage_mask.getdata()

    background = 0
    bands = 0
    stains = 0
    dots = 0
    scratches = 0

    for all_mask_pixel in damage_mask_data:
        if all_mask_pixel == (255, 0, 0):  # red
            bands += 1
        elif all_mask_pixel == (0, 255, 0):  # green
            stains += 1
        elif all_mask_pixel == (0, 0, 255):  # blue
            dots += 1
        elif all_mask_pixel == (255, 255, 255):  # white
            scratches += 1
        else:
            background += 1

    print("bands = ", str(bands)
          + "\nstains = " + str(stains)
          + "\ndots = " + str(dots)
          + "\nscratches = " + str(scratches)
          + "\nbackground = " + str(background))

    score = (bands + stains + dots + scratches) / (background + bands + stains + dots + scratches) * 100
    score = str(round(score, 2))
    return score


# TODO does not work yet, needs fixing
def calculate_weighted_damage_score(damage_mask):
    all_pixels = distance_from_center(damage_mask)
    print(all_pixels)
    background = 0
    bands = 0
    stains = 0
    dots = 0
    scratches = 0

    for x, y in all_pixels:
        coordinate = x, y
        pixel = damage_mask.getpixel(coordinate)
        if pixel == (255, 0, 0):  # red
            bands += 1
        elif pixel == (0, 255, 0):  # green
            stains += 1
        elif pixel == (0, 0, 255):  # blue
            dots += 1
        elif pixel == (255, 255, 255):  # white
            scratches += 1
        else:
            background += 1

    print("bands = ", str(bands)
          + "\nstains = " + str(stains)
          + "\ndots = " + str(dots)
          + "\nscratches = " + str(scratches)
          + "\nbackground = " + str(background))

    score = (bands + stains + dots + scratches) / (background + bands + stains + dots + scratches) * 100
    score = str(round(score, 2))
    return score


# distance from the center
def distance_from_center(image):
    center = np.array([(image.size[0]) / 2, (image.size[1]) / 2])
    distances = np.linalg.norm(np.indices(image.size) - center[:, None, None] + 0.5, axis=0)

    return distances

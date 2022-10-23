import numpy as np
import albumentations as A


def mask_to_consider(lst):
    rgb_values = []
    class_diz = {'background': [0, 0, 0],   # black
                 'bands': [255, 0, 0],      # red
                 'stains': [0, 255, 0],     # green
                 'scratches': [0, 0, 255],  # blue
                 'dots': [255, 255, 255]    # white
                }
    
    for cls in lst:
        if cls in class_diz:
            rgb_values.append(clas_diz[cls])
    
    return np.array(rgb_values)


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
                new_arr[i, j, :] = 0     #useless, just to be sure
            elif seg_map[i, j] == 1:
                new_arr[i, j, 0] = 255
            elif seg_map[i, j] == 2:
                new_arr[i, j, 1] = 255
            elif seg_map[i, j] == 3:
                new_arr[i, j, 2] = 255
            elif seg_map[i, j] == 4:
                new_arr[i, j, :] = 255
    
    return new_arr

import torch
import numpy as np
import os
import albumentations as A
import cv2
from model import load_model
from utils import mask_to_consider, from_segmentation_to_mask, get_preprocessing
from custom_dataset import SegmentationDataset, ValidationDataset

# Specific .pth model to load
load_from = 'C:/Users/shake/Downloads/Compressed/Public-AI-Challenge-Progetto-Caproni-Ludovico/Public-AI-Challenge-Progetto-Caproni-Ludovico/DeepLabV3Plus/models/EP0008~1.PTH'

# Images for which to predict a mask
root_dir = 'C:/Users/shake/Downloads/Compressed/Public-AI-Challenge-Progetto-Caproni-Ludovico/Public-AI-Challenge-Progetto-Caproni-Ludovico/DeepLabV3Plus/test_topredict'
# Folder where to store predicted masks
results_folder = 'predicted'

if not os.path.exists(results_folder):
    os.mkdir(results_folder)

classes = ['background', 'bands', 'stains', 'dots', 'scratches']
select_class_rgb_values = mask_to_consider(classes)

DEVICE = torch.device('cpu')

model = load_model(DEVICE, n_classes=len(classes), load_from=load_from)

# ValidationDataset loads only images (not masks) from the provided root_dir
caproni_dataset = ValidationDataset(root_dir=root_dir,
                                    preprocessing=None,
                                    class_rgb_values=select_class_rgb_values)

for idx in range(len(caproni_dataset)):
        image = caproni_dataset[idx]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask = from_segmentation_to_mask(pred_mask)
        image = np.transpose(image, (1, 2, 0))
        cv2.imwrite(f'{results_folder}/pred_{idx}.png', np.hstack([image, pred_mask]))

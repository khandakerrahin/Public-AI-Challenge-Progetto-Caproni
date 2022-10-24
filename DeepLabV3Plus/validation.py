import torch
import numpy as np
import os
import albumentations as A
import cv2
from glob import glob
from model import load_model
import segmentation_models_pytorch as smp
from utils import mask_to_consider, from_segmentation_to_mask, get_preprocessing
from custom_dataset import ValidationDataset


caproni_dir = '/Public-AI-Challenge-Progetto-Caproni/DeepLabV3Plus/caproni_topredict'
prev_epochs = '/Public-AI-Challenge-Progetto-Caproni/DeepLabV3Plus/epochs_model'
results_folder = '/Public-AI-Challenge-Progetto-Caproni/DeepLabV3Plus/caproni_results'
if not os.path.exists(results_folder):
    os.mkdir(results_folder)
    
classes = ['background', 'bands', 'stains', 'scratches', 'dots']
select_class_rgb_values = mask_to_consider(classes)

DS_MEAN = (0.485, 0.456, 0.406)
DS_STD = (0.229, 0.224, 0.225)

test_transform = A.Compose([
    A.Resize(height=224, width=224),
    #A.Normalize(mean=DS_MEAN,
    #            std=DS_STD)
])

DEVICE = torch.device('cuda')

model = load_model(device=DEVICE, load_best=True, epochs_dir=prev_epochs)

caproni_dataset = ValidationDataset(caproni_dir,
                                    transform=test_transform,
                                    class_rgb_values=select_class_rgb_values)

if __name__ == '__main__':
    for idx in range(len(caproni_dataset)):
        image = caproni_dataset[idx]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        x_tensor = x_tensor.type(torch.cuda.FloatTensor)    # This further step is required because no preprocessing is done
        x_tensor = torch.permute(x_tensor, (0, 3, 1, 2))    # and also this
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask = from_segmentation_to_mask(pred_mask)
        cv2.imwrite(f'{results_folder}/pred_{idx}.png', np.hstack([image, pred_mask]))

import torch
import numpy as np
import albumentations as A
import cv2
from glob import glob
from model import load_model
import os

root_dir = '/Public-AI-Challenge-Progetto-Caproni/DeepLabV3Plus/damage_dataset_splitted'
model_path = '/Public-AI-Challenge-Progetto-Caproni/DeepLabV3Plus/epochs_model/specific_model.pth'

results_folder = '/Public-AI-Challenge-Progetto-Caproni/DeepLabV3Plus/caproni_results'
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

DEVICE = torch.device('cuda')

classes = ['background', 'bands', 'stains', 'scratches', 'dots']

select_class_rgb_values = mask_to_consider(classes)

DS_MEAN = (0.485, 0.456, 0.406)
DS_STD = (0.229, 0.224, 0.225)

test_transform = A.Compose([
    A.Resize(height=224, width=224),
    #A.Normalize(mean=DS_MEAN,
    #            std=DS_STD)
])

model = load_model(DEVICE, force_model=model_path, n_classes=5)

test_dataset = DamageDataset(root_dir,
                             class_rgb_values=select_class_rgb_values,
                             transform=test_transform,
                             preprocessing=False,
                             train=False)

if __name__ == '__main__':
    for idx in range(len(test_dataset)):
        image, mask = test_dataset[idx]
        image, mask = np.transpose(image, (1, 2, 0)), np.transpose(mask, (1, 2, 0))
        mask = from_segmentation_to_mask(reverse_one_hot(mask))
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        x_tensor = x_tensor.type(torch.cuda.FloatTensor)
        x_tensor = torch.permute(x_tensor, (0, 3, 1, 2))
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask = from_segmentation_to_mask(pred_mask)
        cv2.imwrite(f"{results_folder}/pred_{idx}.png", np.hstack([image, mask, pred_mask]))

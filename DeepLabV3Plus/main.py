import albumentations as A
import numpy as np
import torch
import warnings
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from utils import mask_to_consider, get_preprocessing
from custom_dataset import DamageDataset
from segmentation_models_pytorch.utils import train, metrics
from model import load_model
import os


root_dir = '/Public-AI-Challenge-Progetto-Caproni/DeepLabV3Plus/damage_dataset_splitted'
prev_epochs = '/Public-AI-Challenge-Progetto-Caproni/DeepLabV3Plus/epochs_model'

if not os.path.exists(prev_epochs):
    os.mkdir(prev_epochs)
    
classes = ['background', 'bands', 'stains', 'scratches', 'dots']

select_class_rgb_values = mask_to_consider(classes)

DS_MEAN = (0.485, 0.456, 0.406)
DS_STD = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    #A.Normalize(mean=DS_MEAN,
    #            std=DS_STD),
    A.OneOf(
        [A.HorizontalFlip(p=1),
         A.VerticalFlip(p=1),
         A.RandomRotate90(p=1)], p=0.75)
])

test_transform = A.Compose([
    A.Normalize(mean=DS_MEAN,
                std=DS_STD)
])

#preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name='resnet101',
#                                                     pretrained='imagenet')

train_dataset = DamageDataset(root_dir=root_dir,
                              class_rgb_values=select_class_rgb_values,
                              transform=train_transform,
                              preprocessing=None,# get_preprocessing(preprocessing_fn),
                              train=True)

test_dataset = DamageDataset(root_dir=root_dir,
                             class_rgb_values=select_class_rgb_values,
                             transform=None,
                             preprocessing=None,# get_preprocessing(preprocessing_fn),
                             train=False)

# HyperParams

N_EPOCHS = 10
N_CLASSES = 5 # background, bands, stains, scratches, dots
DEVICE = torch.device('cuda')
LR = 1e-4

# Load model from prev epochs (after having fine-tuned)
#model = load_model(DEVICE, load_best=True, epochs_dir=prev_epochs, n_classes=5)
#model.to(DEVICE)

# Initialize model for the first time (fine-tuning)
model = load_model(DEVICE, load_best=False, n_classes=5)
model.to(DEVICE)

loss = smp.losses.FocalLoss('multilabel', alpha=None)

loss.__name__ = 'focal_loss'

chosen_metrics = [metrics.IoU(threshold=0.5)]

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=LR)])

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5)

train_epoch = train.TrainEpoch(model,
                               loss=loss,
                               metrics=chosen_metrics,
                               optimizer=optimizer,
                               device=DEVICE,
                               verbose=True)

test_epoch = train.ValidEpoch(model,
                              loss=loss,
                              metrics=chosen_metrics,
                              device=DEVICE,
                              verbose=True)

if __name__ == '__main__':
    warnings.filterwarnings('ignore') # Not working for libpng warnings =(
    
    for epoch in range(N_EPOCHS):
        print(f'\nEPOCH: {epoch}')
        train_logs = train_epoch.run(train_loader)
        test_logs = test_epoch.run(test_loader)
        torch.save(model, f"{prev_epochs}/epoch_{epoch}-testIoU_{test_logs['iou_score']}.pth")

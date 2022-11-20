import torch
import warnings
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from utils import mask_to_consider, get_preprocessing
from custom_dataset import SegmentationDataset
from segmentation_models_pytorch.utils import train, metrics
from model import load_model
import os
# import torch as t
# print(t.__version__)
# print(torch.version.cuda)
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# torch.cuda.empty_cache()

# torch.cuda.memory_summary(device=None, abbreviated=False)
# root_dir = 'C:/Users/shake/Downloads/Compressed/Public-AI-Challenge-Progetto-Caproni-Ludovico/Public-AI-Challenge-Progetto-Caproni-Ludovico/DeepLabV3Plus/damage_dataset_splitted'
root_dir = 'C:/Users/shake/Downloads/Compressed/Public-AI-Challenge-Progetto-Caproni-Ludovico/Public-AI-Challenge-Progetto-Caproni-Ludovico/DeepLabV3Plus/damage_dataset_v4'
prev_epochs = 'C:/Users/shake/Downloads/Compressed/Public-AI-Challenge-Progetto-Caproni-Ludovico/Public-AI-Challenge-Progetto-Caproni-Ludovico/DeepLabV3Plus/models'

if not os.path.exists(prev_epochs):
    os.mkdir(prev_epochs)

classes = ['background', 'bands', 'stains', 'scratches']

select_class_rgb_values = mask_to_consider(classes)


#preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name='resnet101',
#                                                     pretrained='imagenet')


train_dataset = SegmentationDataset(root_dir=root_dir,
                              class_rgb_values=select_class_rgb_values,
                              transform=None,
                              preprocessing=get_preprocessing(),
                              train=True)

test_dataset = SegmentationDataset(root_dir=root_dir,
                             class_rgb_values=select_class_rgb_values,
                             transform=None,
                             preprocessing=get_preprocessing(),
                             train=False)
torch.cuda.empty_cache()
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8)

# HyperParams
isCudaAvailable = torch.cuda.is_available()
# print("CUDA: ", isCudaAvailable)
DEVICE = torch.device('cuda') if isCudaAvailable else torch.device('cpu')
N_EPOCHS = 10
N_CLASSES = len(classes)
LR = 1e-4


model = load_model(DEVICE, n_classes=N_CLASSES, load_from=None)
model.to(DEVICE)
torch.cuda.empty_cache()
loss = smp.losses.FocalLoss('multilabel')
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
        print(f'\nEPOCH: {epoch}\n')
        train_logs = train_epoch.run(train_loader)
        test_logs = test_epoch.run(test_loader)
        torch.save(model, f"{prev_epochs}/epoch_{epoch}--trainloss{train_logs['focal_loss']}--testloss{test_logs['focal_loss']}--trainiou{train_logs['iou_score']}--testiou{test_logs['iou_score']}.pth")

    

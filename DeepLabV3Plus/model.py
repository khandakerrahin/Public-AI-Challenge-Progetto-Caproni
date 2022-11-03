import segmentation_models_pytorch as smp
import torch


def load_model(device, n_classes, load_from=None):
    if load_from is not None:
        model = torch.load(load_from, map_location=device)
    else:
        model = smp.DeepLabV3Plus(encoder_name='resnet101',
                                  encoder_weights='imagenet',
                                  in_channels=3,
                                  classes=n_classes)
    return model

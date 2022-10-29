import torch
from glob import glob
import segmentation_models_pytorch as smp
import warnings

def load_model(device: torch.device, load_best: bool = False, epochs_dir: str = '', n_classes: int = 5, force_model: str = ''):
    """
    :param device: device in which to send the model (e.g., torch.device('cuda'))
    :param load_best: whether to retrieve the best available model according to its IoU or init a new model
    :param epochs_dir: folder in which to check for the best available model
    :param n_classes: number of classes to predict
    :return: model
    """
    if force_model != '':
        model = torch.load(force_model, map_location=device)
        
    elif load_best:
        if epochs_dir is None:
            raise ValueError('You are trying to load a model from previous epochs, but the folderfrom which to retrieve it is None.')
        else:
            possible_models = glob(f'{epochs_dir}/*.pth')
            if len(possible_models) == 0:
                raise ValueError(f'You have provided "{epochs_dir}" as folder from which to load a model, but there is no model in it.')
            else:
                chosen_model = ('', -1)
                for mod in possible_models:
                    iou = float(mod.split('/')[-1].split('_')[-1].replace('.pth', ''))
                    if iou > chosen_model[-1]:
                        chosen_model = (mod, iou)
                model = torch.load(chosen_model[0], map_location=device)
                
    else:
        if epochs_dir:
            warnings.warn('epochs_dir has been set, but load_best=False: A new model is going to be initialized', UserWarning)
        model = smp.DeepLabV3Plus(encoder_name='resnet101',
                                  encoder_weights='imagenet',
                                  classes=n_classes,
                                  activation='sigmoid')
        
    return model

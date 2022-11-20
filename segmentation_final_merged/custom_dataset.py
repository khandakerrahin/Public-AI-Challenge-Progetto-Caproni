from torch.utils.data import Dataset
from glob import glob
import cv2
from torchvision.transforms import transforms

from utils import one_hot_encode
import torch


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, class_rgb_values, transform=None, preprocessing=None, train=True):
        task = 'train' if train else 'test'
        self.images = glob(f'{root_dir}/{task}/image/*')
        self.images.sort()
        self.masks = glob(f'{root_dir}/{task}/mask/*')
        self.masks.sort()
        self.preprocessing = preprocessing
        self.class_rgb_values = class_rgb_values
        self.transform = transform
        torch.cuda.empty_cache()
        assert len(self.images) == len(self.masks)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.masks[idx]), cv2.COLOR_BGR2RGB)

        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        dim = (224, 224)

        # resize image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)

        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing is not None:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        else:
            image, mask = image.transpose(2, 0, 1).astype('float32'), mask.transpose(2, 0, 1).astype('float32')
        torch.cuda.empty_cache()
        return image, mask

    def __len__(self):
        return len(self.images)


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, preprocessing=None, class_rgb_values=None):
        self.images = glob(f'{root_dir}/*')
        self.images.sort()
        self.preprocessing = preprocessing
        self.transform = transform
        self.class_rgb_values = class_rgb_values

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        # image = Image.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            sample = self.transform(image=image)
            image = sample['image']
        if self.preprocessing is not None:
            sample = self.preprocessing(image=image)
            image = sample['image']
        else:
            image = image.transpose(2, 0, 1).astype('float32')

        return image, self.images[idx]

    def __len__(self):
        return len(self.images)

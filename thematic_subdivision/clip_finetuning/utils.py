from PIL import Image
from glob import glob
import clip


class CustomDataset:
    def __init__(self, image_dir, preprocess):
        img_dir = glob(image_dir + '**/*')
        self.image_path = img_dir
        self.label = [path.split('/')[-2] for path in self.image_path]

        classes = list(set(self.label))
        self.classes_to_idx = {classes[i]: i for i in range(len(classes))}
        self.preprocess = preprocess

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx]).convert("RGB"))
        label = self.label[idx]
        path = self.image_path[idx]
        class_ids = self.classes_to_idx[label]
        return image, label, class_ids, path


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

from PIL import Image
from glob import glob
import clip


class CustomDataset:
    def __init__(self, image_dir, preprocess):
        img_dir = glob(image_dir + '**/*')
        self.image_path = img_dir
        self.title = clip.tokenize([path.split('/')[-2] for path in img_dir])
        self.preprocess = preprocess

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx]).convert("RGB"))
        title = self.title[idx]
        path = self.image_path[idx]
        return image, title, path



def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

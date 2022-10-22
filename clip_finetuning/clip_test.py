import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load fine-tuned model
model, preprocess = clip.load("ViT-B/32", device=device)
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint)


class_names = ['newspaper', 'airplane', 'object', 'vehicle', 'building', 'figure', 'people', 'landscape']


dataset = CustomDataset('./data/test/', preprocess)   # folder without labels

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# CLIP prediction for test images
out = {k: [] for k in class_names}

with torch.no_grad():
    for i, data in enumerate(dataloader):
        image, label, pth = data
        image = image.to(device)

        image_features = model.encode_image(image).float()
        text_features = model.encode_text(clip.tokenize(class_names))

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        pred_class_idx = torch.argmax(similarity, dim=1)
        pred_class_names_batch = [class_names[i] for i in pred_class_idx]

        for j, p in enumerate(pth):
            out[pred_class_names_batch[j]].append(p)


# plot image with predicted label
for label in out:
    images = out[label]
    for im in images:
        plt.imshow(plt.imread(im), cmap='gray')
        plt.title(label)
        plt.axis('off')
        plt.show()

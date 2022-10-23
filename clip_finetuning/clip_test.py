import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

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

p_bar = tqdm(dataloader)
with torch.no_grad():
    for i, data in enumerate(p_bar):
        image, label, class_ids, pth = data
        image = image.to(device)

        image_features = model.encode_image(image).float()
        text_features = model.encode_text(clip.tokenize(class_names))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        pred_class_idx = torch.argmax(similarity, dim=1)
        pred_class_names_batch = [class_names[i] for i in pred_class_idx][0]
        out[pred_class_names_batch].append(pth[0])



# plot image with predicted label
for label in out:
    images = out[label]
    for i, im in enumerate(images):
        if i > 2:
            break
        plt.imshow(plt.imread(im), cmap='gray')
        plt.title(label)
        plt.axis('off')
        plt.show()

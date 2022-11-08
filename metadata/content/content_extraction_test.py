import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

import torch
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification
from glob import glob
from PIL import Image
import torch.nn as nn
from torch.nn.functional import sigmoid

model_path = '/path_to_checkpoint'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path, do_resize=False)
model = ViTForImageClassification.from_pretrained(model_path)


def all_labels(df):
    labels = []
    for i in range(len(df)):
        cnt = df.content[i].split(',')
        for x in cnt:
            if x not in labels:
                labels.append(x)
    return sorted(labels)


def get_lab_mask(labels, content):
    return np.array(list({i: 1 if i in content.split(',') else 0 for i in labels}.values()))


df = pd.read_csv('/path_to_csv_with_path_and_content/')

labs = all_labels(df)
df['content_mask'] = df.content.apply(lambda x: get_lab_mask(labs, x))

images = list(df.loc[df.img_path.str.contains('test')].img_path)
labels = list(df.loc[df.img_path.str.contains('test')].content_mask)


#
corr = 0
n = 0
for i, image in enumerate(tqdm(images)):
    y_true = torch.tensor(labels[i])
    im = Image.open(image).convert("RGB").resize((224, 224))
    encoding = feature_extractor([im], return_tensors='pt')
    output = model(**encoding)
    y_pred = output.logits.sigmoid()
    true_labels = [model.config.id2label[j] for j in np.where(y_true == 1)[0]]
    pred_labels = [model.config.id2label[j] for j in np.where(y_pred[0] >= 0.5)[0]]
    if np.sum(((y_pred >= 0.5) == y_true).numpy()) == len(labs):
        corr += 1
    if len(pred_labels) == 0:
        pred_labels = model.config.id2label[y_pred.argmax(-1).item()]
    if true_labels != pred_labels and n <= 10:
        plt.imshow(im)
        plt.title(f"true: {true_labels} \npred: {pred_labels}")
        plt.show()
        n += 1

print(f"accuracy: {round(corr/len(images), 4)}")

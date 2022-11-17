import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification, \
    AutoFeatureExtractor, SwinForImageClassification
from PIL import Image


model_path = './results/checkpoint-n'     # n is the checkpoint number
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, do_resize=False)
model = SwinForImageClassification.from_pretrained(model_path)


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


# csv file contains: image paths, label and content columns
df = pd.read_csv('./path_to_csv')

labs = all_labels(df)
df['content_mask'] = df.content.apply(lambda x: get_lab_mask(labs, x))

images = list(df.loc[df.img_path.str.contains('test')].img_path)
labels = list(df.loc[df.img_path.str.contains('test')].content_mask)


#
corr = {k: {"correct": 0, "total": 0} for k in labs}
for i, image in enumerate(tqdm(images)):
    y_true = torch.tensor(labels[i])
    im = Image.open(image).convert("RGB").resize((384, 384))
    encoding = feature_extractor([im], return_tensors='pt')
    output = model(**encoding)
    y_pred = output.logits.sigmoid()
    true_labels = [model.config.id2label[j] for j in np.where(y_true == 1)[0]]
    pred_labels = [model.config.id2label[j] for j in np.where(y_pred[0] >= 0.5)[0]]
    if len(pred_labels) == 0:
        pred_labels = [model.config.id2label[y_pred.argmax(-1).item()]]
    # if np.sum(((y_pred >= 0.5) == y_true).numpy()) == len(labs):
    #     corr += 1
    for pred in pred_labels:
        if pred in true_labels:
            corr[pred]['correct'] += 1
        corr[pred]['total'] += 1


# print(f"accuracy: {round(corr/len(images), 4)}")
for k in corr:
    if corr[k]['total'] != 0:
        print(f"accuracy for label {k}: {corr[k]['correct'] / corr[k]['total']}")

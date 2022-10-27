import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset, load_metric
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import torch
import numpy as np
from glob import glob
from PIL import Image


model_path = "./results"

feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path)

images = glob('../clip_finetuning/data/test/**/*')
random.shuffle(images)

correct = 0
n = 0
for i, image in enumerate(tqdm(images)):
    y_true = image.split('/')[-2]
    im = Image.open(image).convert("RGB").resize((224, 224))
    encoding = feature_extractor([im], return_tensors='pt')
    output = model(**encoding)
    idx = output.logits.argmax(-1).item()
    label = model.config.id2label[idx]
    if label == y_true:
        correct += 1
    n += 1

print(f"correct {correct} over {n}")

import os
import shutil
from glob import glob
from PIL import Image
import datasets
from datasets import load_dataset, load_metric
from transformers import ViTFeatureExtractor, ViTForImageClassification, AutoFeatureExtractor, \
    SwinForImageClassification
import torch
import numpy as np
from tqdm import tqdm


class Classify:
    def __init__(self, input_folder, task='thematic_subdivision'):
        self.input_folder = input_folder
        self.task = task
        if task == "thematic_subdivision":
            # self.model_folder =
            self.feature_extractor = ViTFeatureExtractor.from_pretrained("./checkpoints/classification_checkpoint")
            self.model = ViTForImageClassification.from_pretrained("./checkpoints/classification_checkpoint")
        elif task == "damage_assessment":
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("./checkpoints/damage_checkpoint")
            self.model = SwinForImageClassification.from_pretrained("./checkpoints/damage_checkpoint")

    def classify(self):
        output_dir = os.path.join(self.input_folder, f"{self.task}_result")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=False, onerror=None)

        os.makedirs(output_dir)

        images = glob(os.path.join(self.input_folder, "*.jpg"))

        for l in [k for k in self.model.config.label2id]:
            os.makedirs(os.path.join(output_dir, l))

        out = {}
        for i, image in enumerate(tqdm(images)):
            im = Image.open(image).convert("RGB").resize((224, 224))
            encoding = self.feature_extractor([im], return_tensors='pt')
            output = self.model(**encoding)
            idx = output.logits.argmax(-1).item()
            label = self.model.config.id2label[idx]
            if label not in out:
                out[label] = []
            out[label].append(image)

        for k in out:
            images = out[k]
            dest = os.path.join(output_dir, k)
            c = [shutil.copy(im, os.path.join(dest, im.split('/')[-1])) for im in images]

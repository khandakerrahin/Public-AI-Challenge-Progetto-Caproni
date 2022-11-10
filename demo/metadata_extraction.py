import os
from glob import glob
import pandas as pd
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np
from tqdm import tqdm


class MetadataExtraction:
    def __init__(self, input_folder, output_folder):
        self.images = sorted(glob(os.path.join(input_folder, "*.jpg")))
        self.output_folder = os.path.join(output_folder, "metadata_results.csv")
        self.classification_model = "./checkpoints/classification_checkpoint"
        self.content_model = "./checkpoints/content_checkpoint"
        # self.caption_model = None
        self.output = {"image_path": self.images,
                       "subject": [],
                       "content": [],
                       # "description": [],
                       }

    def get_subject(self):
        feature_extractor = ViTFeatureExtractor.from_pretrained(self.classification_model)
        model = ViTForImageClassification.from_pretrained(self.classification_model)
        labels = []
        for i, image in enumerate(tqdm(self.images)):
            im = Image.open(image).convert("RGB")
            encoding = feature_extractor([im], return_tensors='pt')
            output = model(**encoding)
            idx = output.logits.argmax(-1).item()
            label = model.config.id2label[idx]
            labels.append(label)
        return labels

    def get_content(self):
        feature_extractor = ViTFeatureExtractor.from_pretrained(self.content_model)
        model = ViTForImageClassification.from_pretrained(self.content_model)

        contents = []
        for i, image in enumerate(tqdm(self.images)):
            im = Image.open(image).convert("RGB")
            encoding = feature_extractor([im], return_tensors='pt')
            output = model(**encoding)
            logits = output.logits.sigmoid()
            predicted_labels = [model.config.id2label[j] for j in np.where(logits[0] >= 0.5)[0]]
            if len(predicted_labels) == 0:
                predicted_labels = [model.config.id2label[logits.argmax(-1).item()]]
            contents.append(", ".join(predicted_labels))
        return contents

    def get_description(self):
        pass

    def get_metadata(self):
        self.output['subject'] = self.get_subject()
        self.output['content'] = self.get_content()
        data = pd.DataFrame(self.output)
        data.to_csv(self.output_folder, index=False)


import os
from glob import glob
import pandas as pd
import torch.cuda
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification, \
    AutoFeatureExtractor, SwinForImageClassification, AutoTokenizer, VisionEncoderDecoderModel,\
    ViltProcessor, ViltForQuestionAnswering
import numpy as np
from tqdm import tqdm


class MetadataExtraction:
    def __init__(self, input_folder, output_folder):
        self.images_path = sorted(glob(os.path.join(input_folder, "*.jpg")))
        self.images = [Image.open(im).convert("RGB").resize((224, 224)) for im in self.images_path]
        self.output_folder = os.path.join(output_folder, "metadata_results.csv")
        self.classification_model = "./checkpoints/classification_checkpoint"
        self.content_model = "./checkpoints/content_checkpoint"
        self.caption_model = "nlpconnect/vit-gpt2-image-captioning"
        self.damage_model = "./checkpoints/damage_checkpoint"
        self.output = {"image_path": self.images_path}

    def get_subject(self):
        feature_extractor = ViTFeatureExtractor.from_pretrained(self.classification_model)
        model = ViTForImageClassification.from_pretrained(self.classification_model)
        labels = []
        for i, image in enumerate(tqdm(self.images)):
            encoding = feature_extractor([image], return_tensors='pt')
            output = model(**encoding)
            idx = output.logits.argmax(-1).item()
            label = model.config.id2label[idx]
            labels.append(label)
        # return labels
        self.output['subject'] = labels

    def get_content(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.content_model)
        model = SwinForImageClassification.from_pretrained(self.content_model)

        contents = []
        for i, image in enumerate(tqdm(self.images)):
            encoding = feature_extractor([image], return_tensors='pt')
            output = model(**encoding)
            logits = output.logits.sigmoid()
            predicted_labels = [model.config.id2label[j] for j in np.where(logits[0] >= 0.5)[0]]
            if len(predicted_labels) == 0:
                predicted_labels = [model.config.id2label[logits.argmax(-1).item()]]
            contents.append(", ".join(predicted_labels))
        # return contents
        self.output['content'] = contents

    def get_description(self):
        model = VisionEncoderDecoderModel.from_pretrained(self.caption_model)
        feature_extractor = ViTFeatureExtractor.from_pretrained(self.caption_model)
        tokenizer = AutoTokenizer.from_pretrained(self.caption_model)
        gen_kwargs = {"max_length": 20, "num_beams": 4}
        captions = []
        for i, image in enumerate(tqdm(self.images)):
            im = feature_extractor(image, return_tensors='pt').pixel_values
            output_ids = model.generate(im, **gen_kwargs)
            pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            captions += [p.strip() for p in pred]
        # return captions
        self.output['description'] = captions

    def get_damage(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.damage_model)
        model = SwinForImageClassification.from_pretrained(self.damage_model)
        damage_types = []
        for i, image in enumerate(tqdm(self.images)):
            encoding = feature_extractor([image], return_tensors='pt')
            output = model(**encoding)
            idx = output.logits.argmax(-1).item()
            damage = model.config.id2label[idx]
            damage_types.append(damage)
        # return damage_types
        self.output['damage'] = damage_types

    def get_metadata(self):
        # self.output['subject'] = self.get_subject()
        # self.output['content'] = self.get_content()
        # self.output['description'] = self.get_description()
        # self.output['damage'] = self.get_damage()
        data = pd.DataFrame(self.output)
        data.to_csv(self.output_folder, index=False)


import os
from glob import glob
import pandas as pd
import torch.cuda
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification, \
    AutoFeatureExtractor, SwinForImageClassification
from transformers import OFATokenizer, OFAModel
from torchvision import transforms
import numpy as np
from tqdm import tqdm


class MetadataExtraction:
    def __init__(self, input_folder, output_folder):
        self.images_path = sorted(glob(os.path.join(input_folder, "*.jpg")))
        self.images = [Image.open(im).convert("RGB").resize((224, 224)) for im in self.images_path]
        self.output_folder = os.path.join(output_folder, "metadata_results.csv")
        self.classification_model = os.path.join(".", "checkpoint", "classification_checkpoint")
        self.content_model = os.path.join(".", "checkpoints", "content_checkpoint")
        self.caption_model = os.path.join(".", "checkpoints", "caption_checkpoint")
        self.damage_model = os.path.join(".", "checkpoints", "damage_checkpoint")
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
        model = OFAModel.from_pretrained(self.caption_model, use_cache=False)
        tokenizer = OFATokenizer.from_pretrained(self.caption_model)
        patch_resize_transform = transforms.Compose([
            lambda x: x.convert("RGB"),
            transforms.Resize((480, 480), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        q = 'what does the image describe?'

        captions = []
        for i, image in enumerate(tqdm(self.images)):
            inputs = tokenizer([q], return_tensors='pt').input_ids
            img = patch_resize_transform(image).unsqueeze(0)
            gen = model.generate(inputs, patch_images=img, num_beams=5, no_repeat_ngram_size=3)
            caption = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            captions.append(caption)
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
        data = pd.DataFrame(self.output)
        data.to_csv(self.output_folder, index=False)


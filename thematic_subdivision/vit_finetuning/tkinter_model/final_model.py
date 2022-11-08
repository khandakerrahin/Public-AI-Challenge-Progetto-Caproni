import os
import shutil
from glob import glob
from PIL import Image
import datasets
from datasets import load_dataset, load_metric
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import torch
import numpy as np
from tqdm import tqdm


class Train:
    def __init__(self, input_folder, output_folder, model_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_folder = model_folder

        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

        if input_folder is not None:
            self.dataset, self.labels = self.make_dataset()
            self.model = self.make_model()

    def process_example(self, example):
        inputs = self.feature_extractor(example['image'], return_tensors='pt')
        inputs['labels'] = example['labels']
        return inputs

    def transform(self, example_batch):
        # transform PIL into pixel values
        inputs = self.feature_extractor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt')
        inputs['label'] = example_batch['label']
        return inputs

    def collate_fn(self, batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['label'] for x in batch])
        }

    def compute_metrics(self, p):
        metric = load_metric("accuracy")
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    def make_dataset(self):
        f = os.listdir(self.input_folder)
        if 'train' not in f:
            train_ds = load_dataset(self.input_folder, split='train[:80%]')
            validation_ds = load_dataset(self.input_folder, split='train[80%:]')
            dataset = datasets.DatasetDict({'train': train_ds, 'validation': validation_ds})
        else:
            dataset = load_dataset(self.input_folder)
        labels = dataset['train'].features['label'].names
        dataset = dataset.with_transform(self.transform)
        return dataset, labels

    def make_model(self, model_name='google/vit-base-patch16-224-in21k'):
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label={str(i): c for i, c in enumerate(self.labels)},
            label2id={c: str(i) for i, c in enumerate(self.labels)}
        )
        return model

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.model_folder,
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            num_train_epochs=1,
            fp16=False,
            save_steps=100,
            eval_steps=1,
            logging_steps=1,
            learning_rate=2e-4,
            save_total_limit=10,
            remove_unused_columns=False,
            push_to_hub=False,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.feature_extractor,
        )

        train_results = trainer.train()

        trainer.save_model()
        train_metrics = trainer.evaluate(self.dataset['train'])

        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)

        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

        trainer.save_state()

        eval_metrics = trainer.evaluate(self.dataset['validation'])
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    def classify(self):
        output_dir = os.path.join(self.output_folder, "classification_result")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=False, onerror=None)

        os.makedirs(output_dir)

        images = glob(os.path.join(self.output_folder, "*.jpg"))
        feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_folder)
        model = ViTForImageClassification.from_pretrained(self.model_folder)

        for l in [k for k in model.config.label2id]:
            os.makedirs(os.path.join(output_dir, l))

        out = {}
        for i, image in enumerate(tqdm(images)):
            im = Image.open(image).convert("RGB").resize((224, 224))
            encoding = feature_extractor([im], return_tensors='pt')
            output = model(**encoding)
            idx = output.logits.argmax(-1).item()
            label = model.config.id2label[idx]
            if label not in out:
                out[label] = []
            out[label].append(image)

        for k in out:
            images = out[k]
            dest = os.path.join(output_dir, k)
            c = [shutil.copy(im, os.path.join(dest, im.split('/')[-1])) for im in images]

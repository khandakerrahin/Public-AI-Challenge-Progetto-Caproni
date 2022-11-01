from datasets import load_dataset, load_metric
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import torch
import numpy as np

model_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)


def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs


def transform(example_batch):
    # transform PIL into pixel values
    inputs = feature_extractor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt')
    inputs['label'] = example_batch['label']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


dataset = load_dataset('../../clip_finetuning/data/')
labels = dataset['train'].features['label'].names
dataset = dataset.with_transform(transform)

model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)


training_args = TrainingArguments(
  output_dir="results",
  per_device_train_batch_size=16,
  evaluation_strategy="epoch",
  num_train_epochs=10,
  fp16=False,
  save_steps=10,
  eval_steps=10,
  logging_steps=10,
  logging_strategy="epoch",
  learning_rate=2e-4,
  save_total_limit=10,
  save_strategy='epoch',
  remove_unused_columns=False,
  push_to_hub=False,
  load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=feature_extractor,
)

train_results = trainer.train()

trainer.save_model()

trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)

trainer.save_state()

metrics = trainer.evaluate(dataset['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# to get the metrics call trainer.state.log_history

import datasets
import numpy as np
import pandas as pd
from datasets import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer, \
    AutoFeatureExtractor, SwinForImageClassification
import torch
from evaluate import load

model_name = "microsoft/swin-large-patch4-window7-224"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, do_resize=True, size=224)


def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors='pt')
    inputs['label'] = torch.tensor(example['label'])
    return inputs


def transform(example_batch):
    # transform PIL into pixel values
    inputs = feature_extractor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt')
    inputs['label'] = example_batch['label']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch]).float()
    }


metric = load("accuracy")


def compute_metrics(p):
    y_pred = torch.tensor(p.predictions).sigmoid()
    y_true = torch.tensor(p.label_ids)
    accuracy = np.mean(((y_pred >= 0.5) == y_true.byte()).detach().numpy()).sum()
    return {'accuracy': round(accuracy, 4)}


###

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
labels = all_labels(df)
df['content_mask'] = df.content.apply(lambda x: get_lab_mask(labels, x))

diz = {}
for task in ["train", "validation"]:
    ddf = df.loc[df.img_path.str.contains(task)]
    diz[task] = datasets.Dataset.from_dict(
        {"image": ddf.img_path, "label": ddf.content_mask}).cast_column("image", Image())

dataset = datasets.DatasetDict(diz)
dataset = dataset.with_transform(transform)

num_labels = len(labels)
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}

model = SwinForImageClassification.from_pretrained(model_name,
                                                   problem_type="multi_label_classification",
                                                   num_labels=num_labels,
                                                   id2label=id2label,
                                                   label2id=label2id,
                                                   ignore_mismatched_sizes=True)

training_args = TrainingArguments(
    output_dir="./results/vit32",
    per_device_train_batch_size=8,
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
    report_to="none"
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

# # to get the metrics call trainer.state.log_history

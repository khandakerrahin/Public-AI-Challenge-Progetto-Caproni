import random
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification
from glob import glob
from PIL import Image

model_path = "/path/for/fine_tuned_model"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path)

images = glob('../clip_finetuning/data2/test/**/*')
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

print(f"Test accuracy: {round(correct / n, 3)}")

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from utils import *


model, preprocess = clip.load("ViT-B/32", device='cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if device == "cpu":
    model.float()


train_dir = './data/train/'
train_data = CustomDataset(train_dir)
train_dataloader = DataLoader(train_data, batch_size=16)

val_dir = './data/test/'
val_data = CustomDataset(val_dir)
val_dataloader = DataLoader(val_data, batch_size=16)

num_epochs = 10
BATCH_SIZE = 16

model_path = 'best_model.pt'

optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader) * num_epochs)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

best_val_loss = 1e5


for epoch in range(num_epochs):
    step = 0
    train_loss = 0
    model.train()
    pbar = tqdm(train_dataloader, leave=False)

    for batch in pbar:
        step += 1
        optimizer.zero_grad()

        images, texts, pth = batch
        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(images.shape[0]).to(device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        train_loss += total_loss.item()

        if device == "cpu":
            optimizer.step()
            scheduler.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            scheduler.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
    train_loss /= step

    step = 0
    val_loss = 0

    with torch.no_grad():
        model.eval()
        val_pbar = tqdm(val_dataloader, leave=False)

        for batch in val_pbar:
            step += 1

            images, texts, pth = batch
            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(images.shape[0]).to(device)

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            val_loss += total_loss.item()
            val_pbar.set_description(f"validation batchCE: {total_loss.item()}", refresh=True)

        val_loss /= step

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)

    print(f"EPOCH: {epoch+1} - TRAIN_LOSS: {train_loss} - VAL_LOSS: {val_loss}")




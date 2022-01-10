from pathlib import Path
import random

import clip
import numpy as np
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import torchvision
from torchvision import transforms
from tqdm import tqdm

from balanced_batch_sampler import BalancedBatchSampler
from openai_imagenet_template import openai_imagenet_template

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


SAVE_INTERVAL = 10
BATCH_SIZE = 8
NUM_EPOCHS = 1000
weights_path = Path("model_checkpoints")
weights_path.mkdir(exist_ok=True)


#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.requires_grad:
            p.grad.data = p.grad.data.float()


device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  #Must set jit=False for training
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

writer = SummaryWriter()

train_dataset = torchvision.datasets.ImageFolder("data/coco_crops_few_shot/train", transform=preprocess)
train_labels = torch.tensor(train_dataset.targets)
train_sampler = BalancedBatchSampler(train_labels, BATCH_SIZE, 1)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                            #    shuffle=True,
                                            #    batch_size=BATCH_SIZE,
                                               batch_sampler=train_sampler)

val_dataset = torchvision.datasets.ImageFolder("data/coco_crops_few_shot/test", transform=preprocess)
val_labels = torch.tensor(val_dataset.targets)
val_sampler = BalancedBatchSampler(val_labels, BATCH_SIZE, 1)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                            #  batch_size=BATCH_SIZE,
                                             batch_sampler=val_sampler)

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()

for p in model.transformer.parameters():
    p.requires_grad = False
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(
    params, lr=1e-5, weight_decay=0.1)
    # , betas=(0.9, 0.98), eps=1e-6)
    # model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6,
    # weight_decay=0.2)  #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    # weight_decay=0.1)  #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# validation accuracy
values_list, indices_list = [], []
top5_results = []
top1_results = []
acc_top1_list = []
acc_top5_list = []

num_batches_train = len(train_dataloader.dataset)/BATCH_SIZE
num_batches_val = len(val_dataloader.dataset)/BATCH_SIZE

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_train_loss = 0
    j = 0
    print(f"Epoch: {epoch}")
    for batch in tqdm(train_dataloader,total=num_batches_train):
        j+=1
        optimizer.zero_grad()

        list_image, list_txt = batch  #list_images is list of image in numpy array(np.uint8), or list of PIL images

        images = torch.stack([img for img in list_image], dim=0).to(
            device
        )
        # TODO: to use mean of multiple prompts need to pre-compute them.
        # for label_id in list_txt:
        #     embedding = []
        #     for t in openai_imagenet_template:
        #         embedding.append(t(train_dataloader.dataset.classes[label_id]))
        list_txt = [f"a photo of a {train_dataloader.dataset.classes[label_id]}" for label_id in list_txt]
        texts = clip.tokenize(list_txt).to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)

        # total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss = loss_img(logits_per_image, ground_truth)
        # print(f"Loss: {total_loss}")
        epoch_train_loss += total_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    writer.add_scalar("Loss/train", epoch_train_loss / num_batches_train, epoch)

    if epoch % SAVE_INTERVAL == 0:

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, weights_path / f"model_{epoch}.pt")  #just change to your preferred folder/filename
        print(f"Saved weights under model_checkpoint/model_{epoch}.pt.")


    model.eval()
    # validation accuracy
    values_list, indices_list = [], []
    top5_results = []
    top1_results = []
    acc_top1_list = []
    acc_top5_list = []

    num_batches_val = len(val_dataloader.dataset)/BATCH_SIZE
    epoch_val_loss = 0
    for i, batch in enumerate(tqdm(val_dataloader, total=num_batches_val)):
        images, class_ids = batch
        class_ids = class_ids.to(device)
        # image_input = preprocess(image).unsqueeze(0).to(device)
        image_input = images.to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in val_dataloader.dataset.classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            logits_per_image, logits_per_text = model(image_input, text_inputs)
            ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            epoch_val_loss += total_loss

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # values, indices = similarity[0]#.topk(5)

        acc_top1 = torchmetrics.functional.accuracy(similarity, class_ids)
        acc_top5 = torchmetrics.functional.accuracy(similarity, class_ids, top_k=5)
        acc_top1_list.append(acc_top1)
        acc_top5_list.append(acc_top5)
    writer.add_scalar("Loss/val", epoch_val_loss / num_batches_val, epoch)

    print(f"Epoch {epoch} train loss: {epoch_train_loss / num_batches_train}")
    print(f"Epoch {epoch} val loss: {epoch_val_loss / num_batches_val}")

    # compute mean top5 accuracy and top1 accuracy
    mean_top5_accuracy = torch.stack(acc_top5_list).mean().cpu().numpy()
    print(f"Mean Top 5 Accuracy: {mean_top5_accuracy*100}%.")
    writer.add_scalar("Validation Accuracy/Top5", mean_top5_accuracy , epoch)
    mean_top1_accuracy = torch.stack(acc_top1_list).mean().cpu().numpy()
    print(f"Mean Top 1 Accuracy: {mean_top1_accuracy*100}%.")
    writer.add_scalar("Validation Accuracy/Top1", mean_top1_accuracy, epoch)

writer.flush()
writer.close()
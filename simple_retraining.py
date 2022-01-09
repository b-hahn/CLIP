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

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

SAVE_INTERVAL = 1
BATCH_SIZE = 8
NUM_EPOCHS = 1000

#BATCH_SIZE must larger than 1
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# train_dataloader = DataLoader(..., batch_size=BATCH_SIZE)  #Define your own dataloader


#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  #Must set jit=False for training
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

writer = SummaryWriter()

train_dataset = torchvision.datasets.ImageFolder("data/coco_crops_few_shot/train", transform=preprocess)
# train_labels = torch.tensor([i for i in range(len(train_dataset.imgs))])
train_labels = torch.tensor(train_dataset.targets)
train_sampler = BalancedBatchSampler(train_labels, BATCH_SIZE, 1)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                            #    shuffle=True,
                                            #    batch_size=BATCH_SIZE,
                                               batch_sampler=train_sampler)

val_dataset = torchvision.datasets.ImageFolder("data/coco_crops_few_shot/test", transform=preprocess)
# val_labels = torch.tensor([i for i in range(len(val_dataset.imgs))])
val_labels = torch.tensor(val_dataset.targets)
val_sampler = BalancedBatchSampler(val_labels, BATCH_SIZE, 1)
val_dataloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("data/coco_crops_few_shot/test",
                                                                              transform=preprocess),
                                            #  batch_size=BATCH_SIZE,
                                             batch_sampler=val_sampler)

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6,
    weight_decay=0.2)  #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# validation accuracy
values_list, indices_list = [], []
top5_results = []
top1_results = []
acc_top1_list = []
acc_top5_list = []

num_batches_val = len(val_dataloader.dataset)/BATCH_SIZE
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

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # values, indices = similarity[0]#.topk(5)

    acc_top1 = torchmetrics.functional.accuracy(similarity, class_ids)
    acc_top5 = torchmetrics.functional.accuracy(similarity, class_ids, top_k=5)
    # values_list.append(values)
    # indices_list.append(indices)
    acc_top1_list.append(acc_top1)
    acc_top5_list.append(acc_top5)
    # TODO: Could CLIP spit out a super similar word to the one I'm looking for? Then I need some NLP approach to decide whether result is sufficiently synonymous.
    # for cid in class_ids:
    #     top5_results.append(True if cid in indices else False)
    #     top1_results.append(True if cid == indices[0] else False)
    # compute mean top5 accuracy and top1 accuracy
mean_top5_accuracy = torch.stack(acc_top5_list).mean().cpu().numpy()
print(f"Mean Top 5 Accuracy: {mean_top5_accuracy*100}%.")
mean_top1_accuracy = torch.stack(acc_top1_list).mean().cpu().numpy()
print(f"Mean Top 1 Accuracy: {mean_top1_accuracy*100}%.")

num_batches_train = len(train_dataloader.dataset)/BATCH_SIZE
for epoch in range(NUM_EPOCHS):
    epoch_train_loss = 0
    j = 0
    print(f"Epoch: {epoch}")
    for batch in tqdm(train_dataloader,total=num_batches_train):
        # if j > 10:
        #     break
        j+=1
        optimizer.zero_grad()

        list_image, list_txt = batch  #list_images is list of image in numpy array(np.uint8), or list of PIL images

        images = torch.stack([img for img in list_image], dim=0).to(
            device
        )
        list_txt = [f"a photo of a {train_dataloader.dataset.classes[label_id]}" for label_id in list_txt]
        texts = clip.tokenize(list_txt).to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        # print(f"Loss: {total_loss}")
        epoch_train_loss += total_loss
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    writer.add_scalar("Loss/train", epoch_train_loss / num_batches_train, epoch)

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, f"model_{epoch}.pt")  #just change to your preferred folder/filename
    print(f"Saved weights under model_checkpoint/model_{epoch}.pt.")

    # validation accuracy
    values_list, indices_list = [], []
    top5_results = []
    top1_results = []
    acc_top1_list = []
    acc_top5_list = []

    num_batches_val = len(val_dataloader.dataset)/BATCH_SIZE
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

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # values, indices = similarity[0]#.topk(5)

        acc_top1 = torchmetrics.functional.accuracy(similarity, class_ids)
        acc_top5 = torchmetrics.functional.accuracy(similarity, class_ids, top_k=5)
        # values_list.append(values)
        # indices_list.append(indices)
        acc_top1_list.append(acc_top1)
        acc_top5_list.append(acc_top5)
        # TODO: Could CLIP spit out a super similar word to the one I'm looking for? Then I need some NLP approach to decide whether result is sufficiently synonymous.
        # for cid in class_ids:
        #     top5_results.append(True if cid in indices else False)
        #     top1_results.append(True if cid == indices[0] else False)

    print(f"Epoch {epoch} train loss: {epoch_train_loss / num_batches_train}")

    # compute mean top5 accuracy and top1 accuracy
    mean_top5_accuracy = torch.stack(acc_top5_list).mean().cpu().numpy()
    print(f"Mean Top 5 Accuracy: {mean_top5_accuracy*100}%.")
    writer.add_scalar("Validation Accuracy/Top5", mean_top5_accuracy , epoch)
    mean_top1_accuracy = torch.stack(acc_top1_list).mean().cpu().numpy()
    print(f"Mean Top 1 Accuracy: {mean_top1_accuracy*100}%.")
    writer.add_scalar("Validation Accuracy/Top1", mean_top1_accuracy, epoch)

writer.flush()
writer.close()
import random

import clip
import numpy as np
from PIL import Image
import torch
import torchmetrics
import torchvision
from torchvision import transforms
from tqdm import tqdm

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
BATCH_SIZE = 2
NUM_EPOCHS = 10

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

train_dataloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
    "/Users/ben/Downloads/coco_crops_few_shot/train", transform=preprocess),
                                               batch_size=BATCH_SIZE)

val_dataloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
    "/Users/ben/Downloads/coco_crops_few_shot/train", transform=preprocess),
                                               batch_size=BATCH_SIZE)

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6,
    weight_decay=0.2)  #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

num_batches_train = len(train_dataloader.dataset)/BATCH_SIZE
for epoch in range(NUM_EPOCHS):
    j = 0
    print(f"{epoch=}")
    for batch in tqdm(train_dataloader,total=num_batches_train):
        # if j > 10:
        #     break
        j+=1
        optimizer.zero_grad()

        list_image, list_txt = batch  #list_images is list of image in numpy array(np.uint8), or list of PIL images

        images = torch.stack([img for img in list_image], dim=0).to(
            device
        )  # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
        # images = torch.stack([preprocess(Image.fromarray(img.numpy())) for img in list_image], dim=0).to(
        #     device
        list_txt = [f"a photo of a {train_dataloader.dataset.classes[label_id]}" for label_id in list_txt]
        texts = clip.tokenize(list_txt).to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(BATCH_SIZE, dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, f"model_checkpoint/model_{epoch}.pt")  #just change to your preferred folder/filename
    print(f"Saved weights under model_checkpoint/model_{epoch}.pt.")

    # validation accuracy
    values_list, indices_list = [], []
    top5_results = []
    top1_results = []
    acc_top1_list = []
    acc_top5_list = []

    num_batches_val = len(val_dataloader.dataset)/BATCH_SIZE
    for i, batch in enumerate(tqdm(val_dataloader, total=num_batches_val)):
        if i == 0:
            print(f"{batch[0].mean()}")
        # if i > 10:
        #     break
        image, class_ids = batch
        # image_input = preprocess(image).unsqueeze(0).to(device)
        image_input = image
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in val_dataloader.dataset.classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = n .encode_image(image_input)
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
    mean_top5_accuracy = np.mean(acc_top5_list)
    print(f"Mean Top 5 Accuracy: {mean_top5_accuracy*100}%.")
    mean_top1_accuracy = np.mean(acc_top1_list)
    print(f"Mean Top 1 Accuracy: {mean_top1_accuracy*100}%.")

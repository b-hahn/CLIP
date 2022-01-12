from pathlib import Path
import random

import clip
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import torchvision
from tqdm import tqdm

from balanced_batch_sampler import BalancedBatchSampler


def finetune():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    SAVE_INTERVAL = 10
    BATCH_SIZE = 8
    NUM_EPOCHS = 100

    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            if p.requires_grad:
                p.grad.data = p.grad.data.float()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  #Must set jit=False for training
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

    writer = SummaryWriter()
    weights_path = Path("model_checkpoints")
    weights_path.mkdir(exist_ok=True)

    train_dataset = torchvision.datasets.ImageFolder("data/coco_crops_few_shot/train", transform=preprocess)
    train_labels = torch.tensor(train_dataset.targets)
    train_sampler = BalancedBatchSampler(train_labels, BATCH_SIZE, 1)
    # use drop_last = True to ensure each batch contains 8 target classes to choose from.
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_sampler=train_sampler,
                                                drop_last=True)

    test_dataset = torchvision.datasets.ImageFolder("data/coco_crops_few_shot/test", transform=preprocess)
    test_labels = torch.tensor(test_dataset.targets)
    test_sampler = BalancedBatchSampler(test_labels, BATCH_SIZE, 1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_sampler=test_sampler,
                                                drop_last=True)

    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()

    # for p in model.transformer.parameters():
    #     p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params, lr=1e-7, weight_decay=0.0001)

    num_batches_train = len(train_dataloader.dataset)/BATCH_SIZE
    num_batches_val = len(test_dataloader.dataset)/BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        epoch_train_loss = 0
        model.train()
        for batch in tqdm(train_dataloader,total=num_batches_train):
            optimizer.zero_grad()

            images, class_ids = batch

            images = torch.stack([img for img in images], dim=0).to(
                device
            )
            # TODO: to use mean of multiple prompts need to pre-compute them.
            texts = [f"a photo of a {train_dataloader.dataset.classes[label_id]}" for label_id in class_ids]
            texts = clip.tokenize(texts).to(device)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)

            total_train_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_train_loss.backward()
            epoch_train_loss += total_train_loss

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        epoch_train_loss /= num_batches_train
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)

        if epoch % SAVE_INTERVAL == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss,
                }, weights_path / f"model_{epoch}.pt")  #just change to your preferred folder/filename
            print(f"Saved weights under model_checkpoint/model_{epoch}.pt.")

        # Compute test accuracy
        model.eval()
        values_list, indices_list = [], []
        top5_results = []
        top1_results = []
        acc_top1_list = []
        acc_top5_list = []

        num_batches_test = len(test_dataloader.dataset)/BATCH_SIZE
        epoch_test_loss = 0
        for i, batch in enumerate(tqdm(test_dataloader, total=num_batches_test)):
            images, class_ids = batch
            class_ids = class_ids.to(device)

            images = images.to(device)
            texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_dataloader.dataset.classes]).to(device)

            with torch.no_grad():
                # TODO: remove duplicate computation of image and text features
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)

                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)
                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                epoch_test_loss += total_loss

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            acc_top1 = torchmetrics.functional.accuracy(similarity, class_ids)
            acc_top5 = torchmetrics.functional.accuracy(similarity, class_ids, top_k=5)
            acc_top1_list.append(acc_top1)
            acc_top5_list.append(acc_top5)
        writer.add_scalar("Loss/test", epoch_test_loss / num_batches_test, epoch)

        print(f"Epoch {epoch} train loss: {epoch_train_loss / num_batches_train}")
        print(f"Epoch {epoch} test loss: {epoch_test_loss / num_batches_test}")

        # compute mean top5 accuracy and top1 accuracy
        mean_top5_accuracy = torch.stack(acc_top5_list).mean().cpu().numpy()
        print(f"Mean Top 5 Accuracy: {mean_top5_accuracy*100}%.")
        writer.add_scalar("Test Accuracy/Top5", mean_top5_accuracy , epoch)
        mean_top1_accuracy = torch.stack(acc_top1_list).mean().cpu().numpy()
        print(f"Mean Top 1 Accuracy: {mean_top1_accuracy*100}%.")
        writer.add_scalar("Test Accuracy/Top1", mean_top1_accuracy, epoch)

    writer.flush()
    writer.close()

if __name__ == '__main__':
    finetune()
import os


import clip
import numpy as np
import torch
from torchvision.datasets import CIFAR100
import torchvision

# from helsing_dataset import HelsingFewShotDataset

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
# helsing_fewshot_test_data = HelsingFewShotDataset(None, "/Users/ben/Downloads/coco_crops_few_shot")
helsing_fewshot_test_data = torchvision.datasets.ImageFolder("/Users/ben/Downloads/coco_crops_few_shot/test")
helsing_fewshot_test_data_iter = iter(helsing_fewshot_test_data)

# Prepare the inputs
# image, class_id = cifar100[3637]
# image, class_id = next(helsing_fewshot_test_data_iter)
values_list, indices_list = [], []
top5_results = []
top1_results = []

for i, (image, class_id) in enumerate(helsing_fewshot_test_data_iter):
    # if i > 100:
    #     break
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in helsing_fewshot_test_data.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    values_list.append(values)
    indices_list.append(indices)
    # TODO: Could CLIP spit out a super similar word to the one I'm looking for? Then I need some NLP approach to decide whether result is sufficiently synonymous.
    top5_results.append(True if class_id in indices else False)
    top1_results.append(True if class_id == indices[0] else False)


# compute mean top5 accuracy and top1 accuracy
mean_top5_accuracy = np.mean(top5_results)
print(f"Mean Top 5 Accuracy: {mean_top5_accuracy*100}%.")
mean_top1_accuracy = np.mean(top1_results)
print(f"Mean Top 1 Accuracy: {mean_top1_accuracy*100}%.")



# Print the result
# print("\nTop predictions:\n")
# for value, index in zip(values, indices):
#     print(f"{helsing_fewshot_test_data.classes[index]:>16s}: {100 * value.item():.2f}%")

# code adapted from inference_notebook.ipynb - only added args reading
import json
import os
import random
import sys
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch

# Import the nn module from torch.nn
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from torchvision import models, transforms


def load_model(checkpoint_path, num_classes):
    model_name = "efficientnet_b6"
    model = getattr(models, model_name)(weights=None).to(device)

    model.classifier = nn.Linear(2304, num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)

    filtered_state_dict = {
        k: v for k, v in state_dict.items() if k in model.state_dict()
    }

    model.load_state_dict(filtered_state_dict, strict=False)

    model.eval()
    return model


def preprocess_image(image_path, mean, std):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image


def predict(model, image, device):
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    return outputs


def postprocess_predictions(outputs):
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted_idx = torch.max(outputs, 1)
    predicted_label = predicted_idx.item()
    confidence = probabilities[0, predicted_idx].item()
    return predicted_label, confidence


def load_category_mapping(data_path):
    with open(data_path) as f:
        data = json.load(f)

    annotations = pd.DataFrame.from_dict(data["annotations"])
    categories = pd.DataFrame.from_dict(data["categories"])
    category_mapping = {row["id"]: row["name"] for _, row in categories.iterrows()}

    unique_classes = annotations["category_id"].unique()
    category_to_index = {
        category_id: index for index, category_id in enumerate(unique_classes)
    }

    annotations["mapped_category_id"] = annotations["category_id"].map(
        category_to_index
    )

    return category_to_index, category_mapping, annotations


def get_category_name(prediction_label, annotations, category_mapping):
    category_id = annotations.loc[
        annotations["mapped_category_id"] == prediction_label, "category_id"
    ].values[0]
    predicted_label = category_mapping.get(category_id, "Unknown label")
    return predicted_label


def inference(
    image_path,
    checkpoint_path,
    num_classes,
    data_path,
    mean,
    std,
    device=torch.device("cpu"),
):
    model = load_model(checkpoint_path, num_classes)
    image = preprocess_image(image_path, mean, std)
    outputs = predict(model, image, device)
    predicted_label_id, confidence = postprocess_predictions(outputs)

    category_to_index, category_mapping, annotations = load_category_mapping(data_path)
    predicted_label = get_category_name(
        predicted_label_id, annotations, category_mapping
    )

    return predicted_label, confidence


def show_visualization(image_path, predicted_label, confidence):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(
        f"Predicted Label: {predicted_label}\nConfidence: {confidence:.4f}", fontsize=12
    )
    plt.show()


# Example usage
if __name__ == "__main__":
    # improvement to get model and image from console:
    print(sys.argv[1], sys.argv[2])
    image_path = sys.argv[2]
    checkpoint_path = sys.argv[1]
    num_classes = 216
    data_path = "./data/iwildcam2020_train_annotations.json"
    mean = [2.9317e-09, 2.8943e-09, 2.8340e-09]
    std = [1.3166e-09, 1.3476e-09, 1.4205e-09]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predicted_label, confidence = inference(
        image_path, checkpoint_path, num_classes, data_path, mean, std, device
    )
    print(f"Predicted Label: {predicted_label}, Confidence: {confidence:.4f}")

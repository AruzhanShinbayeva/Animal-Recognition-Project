import streamlit as st
import requests
import numpy as np
import cv2
import torch
import json
import pandas as pd
import os

def crop_black_lines(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = image[y : y + h, x : x + w]
        return cropped_image
    else:
        return image

def input_prep(image: np.ndarray):

    img_np = crop_black_lines(image)
    img_np = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_AREA)
    img_tensor = (
        torch.tensor(np.transpose(img_np, (2, 0, 1)), dtype=torch.float32) / 255.0
    )
    
    mean = torch.tensor([2.9317e-09, 2.8943e-09, 2.8340e-09])
    std = torch.tensor([1.3166e-09, 1.3476e-09, 1.4205e-09])
    
    img_tensor = (img_tensor - mean[:, None, None]) / std[:, None, None]

    return img_tensor.numpy().tolist()

data_path = os.path.join('/app/data', 'iwildcam2020_train_annotations.json')
with open(data_path) as f:
    data = json.load(f)

annotations = pd.DataFrame.from_dict(data["annotations"])
categories = pd.DataFrame.from_dict(data["categories"])

category_mapping = {row['id']: row['name'] for _, row in categories.iterrows()}

unique_classes = annotations["category_id"].unique()
category_to_index = {category_id: index for index, category_id in enumerate(unique_classes)}
annotations["mapped_category_id"] = annotations["category_id"].map(category_to_index)


FASTAPI_URL = "http://fastapi:80/predict"

st.title("Animal Recognition")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if uploaded_file:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        processed_input = input_prep(image)

        try:
            response = requests.post(FASTAPI_URL, json={"image": processed_input})
            response.raise_for_status()
            prediction_index = response.json().get("prediction", "No prediction found")

            category_id = annotations.loc[annotations["mapped_category_id"] == prediction_index, "category_id"].values

            if category_id.size > 0:
                category_id_value = category_id[0]
                predicted_label = category_mapping.get(category_id_value, "Unknown label")
            else:
                predicted_label = "No matching category found"

            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write(f"Prediction Index: {prediction_index}")
            st.write(f"Predicted Label: {predicted_label}")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

        except ValueError as e:
            st.error(f"Failed to decode JSON: {e}")

    else:
        st.write("Please upload an image.")
        
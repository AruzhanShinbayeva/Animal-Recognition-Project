import streamlit as st
import requests
import numpy as np
import cv2
import torch

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
    img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)
    img_tensor = (
        torch.tensor(np.transpose(img_np, (2, 0, 1)), dtype=torch.float32) / 255.0
    )
    return img_tensor.numpy().tolist()


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
            prediction = response.json().get("prediction", "No prediction found")

            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write(f"Prediction: {prediction}")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

        except ValueError as e:
            st.error(f"Failed to decode JSON: {e}")
    else:
        st.write("Please upload an image.")
        
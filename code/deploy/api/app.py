import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from torchvision import models


def create_model():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    model_name = "efficientnet_b6"
    model = getattr(models, model_name)(weights=None).to(device)

    num_classes = 216
    model.classifier = nn.Linear(2304, num_classes)
    state_dict = torch.load("/app/models/model.pt", map_location=device)

    filtered_state_dict = {
        k: v for k, v in state_dict.items() if k in model.state_dict()
    }

    model.load_state_dict(filtered_state_dict, strict=False)

    model.eval()
    return model


try:
    model = create_model()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = FastAPI()


class InputData(BaseModel):
    image: list


@app.post("/predict")
def predict(input_data: InputData):
    img_array = np.array(input_data.image, dtype=np.float32)
    image_tensor = torch.tensor(img_array)

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = torch.argmax(outputs, dim=-1)

    return {"prediction": int(predictions.item())}

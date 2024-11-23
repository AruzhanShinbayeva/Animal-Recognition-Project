from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from peft import get_peft_model, LoraConfig
import os
import torch
import torch.nn as nn
from torchvision import transforms

model_path = os.path.join('/app/models', 'model.pt')

def create_model():
    from transformers import AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window16-256"
    )
    num_classes = 216
    model.classifier = nn.Linear(768, num_classes)
    return model

try:
    base_model = create_model()
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.15,
        target_modules=["query", "value", "key"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base_model, lora_config)
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = FastAPI()

class InputData(BaseModel):
    image: list

@app.post("/predict")
def predict(input_data: InputData):

    img_array = np.array(input_data.image, dtype=np.float32)
    img_tensor = torch.tensor(img_array).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        predictions = torch.argmax(outputs.logits, dim=-1)

    return {"prediction": int(predictions.item())}

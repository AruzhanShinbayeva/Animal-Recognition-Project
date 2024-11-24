# Animal Recognition in the Wild: iWildCam2020 Dataset

## About the Project

The **iWildCam2020** dataset presents a unique challenge for animal recognition in the wild, with images captured by automatic cameras deployed in various climatic zones. These cameras document animals under diverse conditions, such as different times of day, varying weather, and across different seasons, which leads to a highly diverse and challenging dataset.

The project aims to develop a deep learning model that can accurately identify and classify animals from these images, despite the following challenges:
- **Significant class imbalance** with 216 animal classes.
- **Varying photo conditions**, such as lighting, weather, and time of day.

## Workflow

The project follows a structured workflow as outlined in the notebook **swin-lora-training.ipynb**, which includes the following stages:

### 1. **Preprocessing the Dataset**
   - Handling the class imbalance.
   - Augmenting data to account for varying photo conditions.
   - Preparing the images for model training.

### 2. **Experimentation with Different Models**
   - **EfficientNet B6**: Used for transfer learning to leverage its ability to scale with different input sizes.
   - **YOLOv11**: Applied for real-time object detection, focusing on accuracy and speed.
   - **SWIN Transformer with LoRA**: A state-of-the-art transformer model that has shown promise in visual tasks, optimized for efficiency with low-rank adaptation (LoRA).

### 3. **Fine-tuning the Best Model**
   - The model with the best initial performance is fine-tuned for higher accuracy and efficiency.
   - Optimizing hyperparameters and training strategies to handle the long training times and stability issues inherent in such a complex dataset.

### 4. **Evaluating Model Performance**
   - Assessing the model's performance using relevant metrics (accuracy, precision, recall, etc.).
   - Evaluating the model under different conditions, including class imbalances, weather, and lighting changes.

### 5. **Deployment**
   - Preparing the final model for deployment in real-world scenarios.
   - Ensuring the model is robust and can handle new, unseen images of animals in various conditions.

## Conclusion
The project aims to achieve high accuracy in classifying animals from diverse and challenging environments, and to develop a stable and efficient model that can be deployed in practical wildlife monitoring systems.


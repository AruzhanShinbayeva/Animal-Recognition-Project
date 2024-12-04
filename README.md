# Animal Recognition Using iWildCam2020 Dataset

This project focuses on the recognition of wild animals captured by camera traps using the iWildCam2020 dataset. It evaluates and implements advanced deep learning models, preprocessing techniques, and data augmentation to address challenges such as class imbalance, variability in lighting, and environmental conditions.

---

## How to run the inference of our project

---

### Install project locally

For running inference locally you need to manually download model weights from our [google-drive model list](https://drive.google.com/drive/folders/1RFIkFalMzkhecT6ncRpkWzSaCHTgLrlr?usp=sharing) and place it to the project root. Then run following to install dependencies:

```cmd
pip install -r requirements.txt
```

Run inference following, where:

- _model_name_ is the name of model weights downloaded earlier
- _path_to_images_ is the path to the image to perform inference on

```cmd
python inference.py model_name path_to_images
```

Example:

```cmd
python inference.py best-effnet-27102024.pt animal_image.jpeg
# Predicted Label: tragelaphus scriptus, Confidence: 0.0047
```

---

### Run usin Docker

Download model from [google-drive model list](https://drive.google.com/drive/folders/1RFIkFalMzkhecT6ncRpkWzSaCHTgLrlr?usp=sharing) and place it to the project-folder/models and name model.pt.

Run inference following:

```cmd
docker-compose -f code/deploy/docker-compose.yml  up --build
```
Now you can access web version for inference at http://localhost:8501

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Models Evaluated](#models-evaluated)
5. [Results and Observations](#results-and-observations)
6. [Conclusion](#conclusion)
7. [Authors](#authors)
8. [References](#references)

---

## Introduction

Recognizing wild animals in camera trap images plays a crucial role in biodiversity monitoring, wildlife conservation, and ecosystem management. However, real-world challenges like poor image quality, occlusions, and class imbalance necessitate robust machine learning solutions. This project explores state-of-the-art architectures for tackling these challenges.

---

## Dataset

### iWildCam2020

The iWildCam2020 dataset contains camera trap images of wild animals captured across various geographic locations, climates, and seasons.

**Key Challenges:**

- Lighting variability (day/night images)
- Black borders with date and time stamps
- Motion blur, occlusions, and inconsistent resolutions
- Significant class imbalance across 216 animal species

---

## Methodology

### Preprocessing Techniques

1. **Day/Night Time Splitting**: Classifies images into daytime and nighttime based on timestamps for better lighting-specific processing.
2. **Dark Image Enhancement**: Uses histogram equalization to improve the visibility of nighttime images.
3. **Black Line Cropping**: Removes black borders or time/date stamps from images.

### Evaluation Metrics

- **F1 Score**
- **Precision**
- **Recall**

These metrics are prioritized over traditional accuracy to reflect model performance on underrepresented classes better.

---

## Models Evaluated

1. **EfficientNet B6**

   - It is highly efficient with fewer parameters and strong classification performance.
   - Robust on high-resolution images but prone to overfitting with small datasets.

2. **YOLOv11 with Classification Head**

   - Real-time performance optimized for object detection.
   - Challenges: Small object detection and class imbalance reduce classification accuracy.

3. **SWIN Transformer with LoRA**
   - Vision Transformer-based model for high-resolution images.
   - Incorporates Low-Rank Adaptation (LoRA) to reduce hardware demands.
   - Prone to overfitting on small datasets and computationally intensive.

---

## Results and Observations

- **EfficientNet B6** emerged as the most balanced model, achieving strong performance when trained on a larger dataset (20,000 images).
- **YOLOv11** exhibited instability due to class imbalance and task mismatch.
- **SWIN Transformer with LoRA** showed promising results but required substantial computational resources and struggled with smaller datasets.

### Improvements

- Increasing the dataset size significantly improved generalization for all models.
- Hard-negative sampling helped balance metrics for underrepresented classes.

---

## Conclusion

EfficientNet B6 was chosen as the optimal model for this task due to its:

- Strong performance across multiple metrics.
- Efficiency with high-resolution images.
- Balanced trade-off between accuracy, recall, and precision.

**Future Work**:

- Advanced data augmentation techniques.
- Enhanced strategies for addressing class imbalance.
- Exploring additional lightweight transformer architectures.

# Traffic Sign Detection and Classification

This project aims to develop a robust system for detecting and classifying Vietnamese traffic signs using state-of-the-art deep learning models. The detection task is handled by YOLOv8, while the classification task is performed using ResNet50. The project is built on the PyTorch framework and leverages datasets from Kaggle.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Datasets](#datasets)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [API Endpoints](#api-endpoints)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Contributors](#contributors)

---

## Project Overview

The project is divided into two main tasks:
1. **Traffic Sign Detection**: Detecting traffic signs in images using YOLOv8.
2. **Traffic Sign Classification**: Classifying detected traffic signs into specific categories using ResNet50.

The system is designed to work with Vietnamese traffic signs and can be extended to other datasets with minimal changes.

---

## Datasets

### 1. **Classification Dataset**
   - **Source**: [Vietnamese Traffic Sign Classification Dataset](https://www.kaggle.com/datasets/tcc3281/vietnametrafficsignclassification)
   - **Description**: Contains labeled images of Vietnamese traffic signs for classification.
   - **Usage**: Used for training the ResNet50 classification model.

### 2. **Detection Dataset**
   - **Source**: [Traffic Signs Detection Dataset](https://www.kaggle.com/datasets/tcc3281/traffic-signs/data)
   - **Description**: Contains images with bounding box annotations for traffic signs.
   - **Usage**: Used for training the YOLOv8 detection model.

---

## Model Architecture

### 1. **YOLOv8 for Detection**
   - **Framework**: Ultralytics YOLOv8
   - **Purpose**: Detect traffic signs in images and return bounding boxes with class labels and confidence scores.

### 2. **ResNet50 for Classification**
   - **Framework**: PyTorch
   - **Modifications**:
     - Fully connected layer modified to match the number of traffic sign classes.
     - Dropout layers added for regularization.
   - **Purpose**: Classify detected traffic signs into specific categories.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.10 or higher
- CUDA (optional, for GPU acceleration)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/tcc3281/Traffic_Sign.git
   cd Traffic_Sign
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets:
   - Classification dataset: [Vietnamese Traffic Sign Classification](https://www.kaggle.com/datasets/tcc3281/vietnametrafficsignclassification)
   - Detection dataset: [Traffic Signs Detection](https://www.kaggle.com/datasets/tcc3281/traffic-signs/data)

4. Place the datasets in the following structure:
   ```
   Traffic_Sign/
   ├── dataset/
   │   ├── classification/
   │   │   ├── train/
   │   │   ├── val/
   │   │   ├── test/
   │   │   └── class_mapping.txt
   │   ├── detection/
   │   │   ├── train/
   │   │   ├── val/
   │   │   ├── test/
   │   │   └── annotations/
   ```

---

## Usage

### 1. **Training**
   - **YOLOv8 Detection**:
     ```bash
     python train_yolo.py --data dataset/detection --weights yolov8n.pt
     ```
   - **ResNet50 Classification**:
     Run the `resnet50-torch.ipynb` notebook to train the classification model.

### 2. **Running the API**
   Start the FastAPI server:
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:8000`.

### 3. **Frontend**
   Open `frontend/index.html` in a browser to interact with the detection and classification system.

---

## API Endpoints

### 1. **/detect**
   - **Method**: POST
   - **Description**: Detects traffic signs in an uploaded image.
   - **Input**: Image file (e.g., `.jpg`, `.png`)
   - **Output**:
     ```json
     {
       "predictions": [
         {
           "class": "P.127-5",
           "confidence": 0.95,
           "bbox": [x1, y1, x2, y2]
         }
       ],
       "image_size": [640, 640],
       "detected_objects": 1,
       "speed": {
         "preprocess": 10.5,
         "inference": 20.3,
         "postprocess": 5.2
       }
     }
     ```

### 2. **/classify**
   - **Method**: POST
   - **Description**: Classifies a traffic sign in an uploaded image.
   - **Input**: Image file (e.g., `.jpg`, `.png`)
   - **Output**:
     ```json
     {
       "classification": {
         "class": "P.127-5",
         "confidence": 0.98
       }
     }
     ```

---

## Results

### Detection
- **Model**: YOLOv8
- **Performance**: Achieved high precision and recall on the detection dataset.

### Classification
- **Model**: ResNet50
- **Performance**: Achieved over 90% accuracy on the classification dataset.

---

## Future Work

1. **Model Improvements**:
   - Experiment with other architectures like EfficientNet or Vision Transformers.
   - Fine-tune YOLOv8 with additional data augmentation techniques.

2. **Dataset Expansion**:
   - Include more diverse traffic sign datasets to improve generalization.

3. **Real-Time Deployment**:
   - Deploy the system on edge devices like Raspberry Pi for real-time traffic sign detection and classification.

---

## Contributors

- **Your Name**: Project Lead
- **Other Contributors**: Add names here

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Repository

The source code for this project is available on GitHub: [Traffic_Sign](https://github.com/tcc3281/Traffic_Sign.git)
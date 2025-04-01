# === Imports ===
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import uvicorn
import time
import torch
from torchvision import models
import torch.nn as nn

# === Global Variables and Configurations ===
NUM_CLASSES = 51
model_yolo = None
model_resnet = None
class_names_resnet = []

yolo_model_path = "model/best.pt"
resnet_model_path = "model/resnet50_best_model_v05.pth"
class_mapping_path = "class_mapping.txt"

# === Utility Functions ===
def load_class_mapping(txt_path):
    """Load class mapping from a file."""
    idx_to_class = {}
    with open(txt_path, 'r') as f:
        for line in f:
            idx, class_name = line.strip().split(':')
            idx_to_class[int(idx)] = class_name.strip()
    return idx_to_class

def process_image(contents: bytes) -> np.ndarray:
    """Process image for YOLO model."""
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read image from input data")
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def process_image_for_resnet(contents: bytes) -> torch.Tensor:
    """Process image for ResNet model."""
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read image")
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

# === Model Loading Functions ===
async def load_yolo_model():
    """Load YOLO model."""
    global model_yolo
    model_yolo = YOLO(yolo_model_path)
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model_yolo.predict(dummy_input, verbose=False)

def load_resnet_model():
    """Load ResNet model."""
    model = models.resnet50(weights=None)
    for name, param in model.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    checkpoint = torch.load(resnet_model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
    num_classes_checkpoint = next((state_dict[key].shape[0] for key in ['fc.weight', 'classifier.weight', 'head.fc.weight'] if key in state_dict), None)
    if num_classes_checkpoint is None:
        num_classes_checkpoint = len(checkpoint.get('classes_order', []))
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes_checkpoint)
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# === FastAPI Application Setup ===
app = FastAPI(
    title="Traffic Sign Detection and Classification API",
    version="1.0",
    docs_url=None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global model_yolo, model_resnet, class_names_resnet
    model_yolo = YOLO(yolo_model_path)
    model_resnet = load_resnet_model()
    class_names_resnet = load_class_mapping(class_mapping_path)
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    _ = model_yolo.predict(dummy_image, verbose=False)
    dummy_tensor = torch.randn(1, 3, 224, 224)
    _ = model_resnet(dummy_tensor)
    yield

app.router.lifespan_context = lifespan

# === API Endpoints ===
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Detect objects in an image using YOLO."""
    try:
        contents = await file.read()
        image = process_image(contents)
        results = model_yolo.predict(image, augment=True)
        predictions = [
            {
                "class": model_yolo.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            }
            for box in results[0].boxes
        ]
        return {
            "predictions": predictions,
            "image_size": results[0].orig_shape,
            "detected_objects": len(results[0].boxes),
            "speed": results[0].speed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Classify an image using ResNet."""
    try:
        contents = await file.read()
        tensor = process_image_for_resnet(contents)
        with torch.no_grad():
            outputs = model_resnet(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probs, dim=1)
        return {
            "classification": {
                "class": class_names_resnet[top_class.item()],
                "confidence": top_prob.item()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

# === Main Entry Point ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import uvicorn
import asyncio
import time
import torch
from torchvision import models

# Biến global cho model_yolo
model_yolo = None
model_resnet = None
class_names_resnet = ['speed limit 20',
                      'speed limit 30',
                      'no truck passing',
                      'no parking',
                      'no horn',
                      'no entry in this direction',
                      'no cars',
                      'speed limit 50',
                      'speed limit 60',
                      'speed limit 70',
                      'speed limit 80',
                      'no entry for all vehicles',
                      'speed limit 100',
                      'speed limit 120',
                      'no passing']

yolo_model_path = "model/best.pt"
resnet_model_path = "model/resnet50_model_v01.pth"


async def load_yolo_model():
    """Tải model YOLO"""
    global model_yolo
    model_yolo = YOLO(yolo_model_path)
    # Warm up model
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model_yolo.predict(dummy_input, verbose=False)


def load_resnet_model():
    """Tải model ResNet với xử lý lỗi state_dict"""
    model = models.resnet50()

    # Thay đổi lớp cuối cùng
    num_classes = 15  # Thay bằng số lớp thực tế
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    # Load state_dict
    state_dict = torch.load(resnet_model_path, map_location='cpu')

    # Xử lý các định dạng state_dict khác nhau
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    # Load vào model
    model.load_state_dict(state_dict, strict=False)  # strict=False để bỏ qua các layer không khớp
    model = model.float()
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng"""
    global model_yolo, model_resnet
    model_yolo = YOLO(yolo_model_path)
    model_resnet = load_resnet_model()

    # Warm up models
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    _ = model_yolo.predict(dummy_image, verbose=False)

    dummy_tensor = torch.randn(1, 3, 224, 224)
    _ = model_resnet(dummy_tensor)

    yield

    # Clean up (nếu cần)


app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    title="Traffic Sign Detection API",
    version="1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


def process_image(contents: bytes) -> np.ndarray:
    """Xử lý ảnh đầu vào tối ưu"""
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không thể đọc ảnh từ dữ liệu đầu vào")

    # Resize và chuẩn hóa ảnh
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@app.post("/detect")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        # Thay đổi 1: Resize ảnh trước khi xử lý
        image = cv2.resize(image, (320, 320))  # Thay đổi kích thước

        # Thay đổi 2: Sử dụng tham số khác cho model
        results = model_yolo.predict(image, augment=True)

        # Tính toán thời gian
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return {
            "predictions": [{
                "class": model_yolo.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            } for box in results[0].boxes],
            "image_size": results[0].orig_shape,
            "detected_objects": len(results[0].boxes),
            "speed": {
                "preprocess": results[0].speed["preprocess"],
                "inference": results[0].speed["inference"],
                "postprocess": results[0].speed["postprocess"]
            },
            "total_time": total_time
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))


def process_image_for_resnet(contents: bytes) -> torch.Tensor:
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image")

    # Resize và chuẩn hóa
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # Quan trọng: float32

    # Chuẩn hóa theo ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    # Chuyển đổi sang tensor và thêm batch dimension
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image.float()  # Đảm bảo kiểu float32


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()

        # Preprocess
        preprocess_start = time.time()
        tensor = process_image_for_resnet(contents)

        # Inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = model_resnet(tensor)

        # Postprocess
        postprocess_start = time.time()
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
        class_name = class_names_resnet[top_class.item()]

        # Tính thời gian
        preprocess_time = (time.time() - preprocess_start) * 1000
        inference_time = (time.time() - inference_start) * 1000
        postprocess_time = (time.time() - postprocess_start) * 1000
        total_time = (time.time() - start_time) * 1000

        return {
            "classification": {
                "class": class_name,
                "confidence": top_prob.item()
            },
            "speed": {
                "preprocess": preprocess_time,
                "inference": inference_time,
                "postprocess": postprocess_time
            },
            "total_time": total_time
        }

    except Exception as e:
        print(e)
        raise HTTPException(500, detail=f"Classification error: {str(e)}")


if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
# python -m http.server 3000 --bind 0.0.0.0
# http://localhost:3000/

import cv2
import numpy as np
import os



def resize_and_save(image_path, label_path, target_size=640, padding_color=(114, 114, 114)):
    """
    Resize ảnh và điều chỉnh bounding boxes, lưu đè lên file gốc.

    Args:
        image_path (str): Đường dẫn đến file ảnh
        label_path (str): Đường dẫn đến file label (.txt)
        target_size (int): Kích thước vuông đầu ra
        padding_color (tuple): Màu padding (BGR)
    """
    # Đọc ảnh và label
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]

    # Đọc tất cả bounding boxes từ file label
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Xử lý resize ảnh
    scale, pad = calculate_padding(orig_w, orig_h, target_size)
    resized_image = apply_resize(image, target_size, scale, pad, padding_color)

    # Xử lý từng bounding box
    new_labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # Bỏ qua dòng lỗi

        # Chuyển đổi tọa độ YOLO (relative) sang absolute
        class_id, x_center_rel, y_center_rel, width_rel, height_rel = map(float, parts)
        x_center = x_center_rel * orig_w
        y_center = y_center_rel * orig_h
        width = width_rel * orig_w
        height = height_rel * orig_h

        # Điều chỉnh tọa độ sau resize
        new_x, new_y, new_w, new_h = adjust_bbox(
            x_center, y_center, width, height,
            scale, pad, target_size
        )

        # Chuyển lại sang định dạng center_xyxy và normalize với kích thước ảnh
        new_x1 = (new_x - new_w / 2) / target_size
        new_y1 = (new_y - new_h / 2) / target_size
        new_x2 = (new_x + new_w / 2) / target_size
        new_y2 = (new_y + new_h / 2) / target_size

        new_labels.append(f"{int(class_id)} {new_x1:.6f} {new_y1:.6f} {new_x2:.6f} {new_y2:.6f}")

    # Ghi đè ảnh và label
    cv2.imwrite(image_path, resized_image)
    with open(label_path, 'w') as f:
        f.write("\n".join(new_labels))


def calculate_padding(orig_w, orig_h, target_size):
    """Tính toán tỉ lệ scale và padding"""
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    return scale, (pad_x, pad_y, new_w, new_h)


def apply_resize(image, target_size, scale, pad, padding_color):
    """Áp dụng resize và padding cho ảnh"""
    pad_x, pad_y, new_w, new_h = pad

    # Resize ảnh
    resized = cv2.resize(image, (new_w, new_h))

    # Thêm padding
    resized = cv2.copyMakeBorder(
        resized,
        pad_y,
        target_size - new_h - pad_y,
        pad_x,
        target_size - new_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=padding_color
    )

    return resized


def adjust_bbox(x_center, y_center, width, height, scale, pad, target_size):
    """Điều chỉnh tọa độ bounding box sau resize"""
    pad_x, pad_y, new_w, new_h = pad

    # Scale tọa độ
    new_x = x_center * scale + pad_x
    new_y = y_center * scale + pad_y
    new_width = width * scale
    new_height = height * scale

    # Đảm bảo không vượt quá biên
    new_x = np.clip(new_x, 0, target_size)
    new_y = np.clip(new_y, 0, target_size)
    new_width = np.clip(new_width, 0, target_size - new_x)
    new_height = np.clip(new_height, 0, target_size - new_y)

    return new_x, new_y, new_width, new_height


def draw_bbox(image_path, label_path):
    """
    Vẽ bounding box lên ảnh và lưu ảnh đã vẽ vào thư mục output.

    Args:
        image_path (str): Đường dẫn đến file ảnh
        label_path (str): Đường dẫn đến file label (.txt)
        output_dir (str): Thư mục lưu ảnh đã vẽ
    """
    # Đọc ảnh và label
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]

    # Đọc tất cả bounding boxes từ file label
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Vẽ từng bounding box lên ảnh
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # Bỏ qua dòng lỗi

        # Chuyển đổi tọa độ YOLO (relative) sang absolute
        class_id, x_center_rel, y_center_rel, width_rel, height_rel = map(float, parts)
        x_center = x_center_rel * orig_w
        y_center = y_center_rel * orig_h
        width = width_rel * orig_w
        height = height_rel * orig_h

        # Tính toán tọa độ góc trên trái và góc dưới phải
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Vẽ bounding box lên ảnh
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Lưu ảnh đã vẽ vào thư mục output
    cv2.imwrite(image_path, image)
# Sử dụng
if __name__ == "__main__":
    image_path = "dataset/Vietnam Traffic Signs/train/images/0001.jpg"
    label_path = "dataset/Vietnam Traffic Signs/train/labels/0001.txt"
    resize_and_save(image_path, label_path, target_size=640)
    draw_bbox(image_path, label_path)
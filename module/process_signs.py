import pandas as pd
import os
from PIL import Image, ImageOps
import shutil
import random


def load_data(folder_path):
    # Danh sách để lưu thông tin ảnh
    image_data = []

    # Duyệt qua các tệp trong thư mục
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Đường dẫn đầy đủ đến tệp ảnh
            img_path = os.path.join(folder_path, filename)
            img_path = img_path.replace("\\", "/")
            # Mở ảnh
            img = Image.open(img_path)
            # Lấy kích thước ảnh
            width, height = img.size
            # Thêm thông tin ảnh vào danh sách
            image_data.append({'filename': filename, 'path': img_path, 'width': width, 'height': height})

    # Tạo DataFrame từ danh sách
    df = pd.DataFrame(image_data)
    return df


def check_path(path):
    '''

    :param path:
    :return: Boolean
    '''
    return os.path.exists(path)


def func(data_frame,nums, path):
    '''

    :param data_frame:
    :return:
    '''
    for i in range(nums):
        new_path = path + str(i)
        if check_path(new_path):
            temp_df = load_data(new_path)
            data_frame = pd.concat([data_frame, temp_df])

    return data_frame

def resize_image(img_path, size):
    '''
    Resize an image to a specific size
    :param img_path: path to the image
    :param size: new size
    :return: resized image
    '''
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)
    img = img.resize(size)
    return img


def split_data(image_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Tạo các thư mục đầu ra nếu chúng chưa tồn tại
    for class_name in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, class_name)
        if os.path.isdir(class_dir):
            train_dir = os.path.join(output_dir, 'images', class_name)
            val_dir = os.path.join(output_dir, 'images', class_name)
            test_dir = os.path.join(output_dir, 'images', class_name)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Lấy danh sách tất cả các ảnh trong thư mục lớp
            images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

            # Xáo trộn các ảnh
            random.shuffle(images)

            # Chia các ảnh
            train_split = int(len(images) * train_ratio)
            val_split = int(len(images) * (train_ratio + val_ratio))

            train_images = images[:train_split]
            val_images = images[train_split:val_split]
            test_images = images[val_split:]

            # Sao chép các ảnh vào các thư mục tương ứng
            for img in train_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, img))

            for img in val_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(val_dir, img))

            for img in test_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, img))

    print("Đã chia ảnh thành các tập images, images, và images cho tất cả các lớp.")

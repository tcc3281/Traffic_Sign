import pandas as pd
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

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

def save_img(path,img):
    #check path

    img.save(path)
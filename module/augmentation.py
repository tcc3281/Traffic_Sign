import Augmentor
import os


def augment_image_augmentor(input_dir, output_dir, num_samples):
    """
    Tăng cường dữ liệu ảnh từ thư mục đầu vào và lưu vào thư mục đầu ra.

    Args:
        input_dir (str): Đường dẫn đến thư mục chứa ảnh gốc.
        output_dir (str): Đường dẫn đến thư mục lưu ảnh đã tăng cường.
    """
    # Kiểm tra input_dir có tồn tại và là thư mục
    if not os.path.isdir(input_dir):
        raise ValueError("input_dir phải là đường dẫn đến thư mục chứa ảnh gốc!")

    # Khởi tạo pipeline
    p = Augmentor.Pipeline(input_dir)

    # Thêm các phép biến đổi (tùy chỉnh tham số tại đây)
    p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
    p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.5)
    p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.5)
    p.zoom_random(probability=0.5, percentage_area=0.85)

    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Thiết lập thư mục đầu ra
    p.output_directory = output_dir
    print(f"Output directory set to: {p.output_directory}")

    # Sinh ảnh: 10 ảnh tổng (tùy chỉnh số lượng)
    p.sample(num_samples)
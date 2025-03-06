from PIL import Image

# Mở hai hình ảnh
image1 = Image.open('i1.jpg')
image2 = Image.open('i2.jpg')

# Lấy kích thước của hai hình ảnh
width1, height1 = image1.size
width2, height2 = image2.size

# Tạo một hình ảnh mới với chiều rộng bằng tổng chiều rộng của hai hình ảnh và chiều cao bằng chiều cao lớn nhất
new_width = width1 + width2
new_height = max(height1, height2)
new_image = Image.new('RGB', (new_width, new_height))

# Dán hai hình ảnh vào hình ảnh mới
new_image.paste(image1, (0, 0))
new_image.paste(image2, (width1, 0))

# Lưu hình ảnh mới
new_image.save('i3.jpg')
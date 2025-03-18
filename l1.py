import cv2

if __name__ == '__main__':

    # Load the image
    # path='dataset/Traffic Sign Detection/test/images/00000_00000_00003_png.rf.d18afc3c9625ffb1974029d3e3762aee.jpg'
    # path='dataset/Traffic Signs/train/images/AUTO_COLLECTED_PICTURE_4_png.rf.609b8a870d6887ad82dea75567e1ded6.jpg'
    path='dataset/detection/train/images/00000_00000_00000_png.rf.55d47572c5980af0892b0c2ada6dae77.jpg'
    # path='dataset/Vietnam Traffic Signs 2/train/images/90.png'
    image = cv2.imread(path)
    if image is None:
        print("Error: Could not open or find the image.")
    else:
        height, width = image.shape[:2]

        # Normalized labels from your message
        normalized_labels = [
            (0.278846, 0.307692, 0.759615, 0.769231),
            (0.269832, 0.304087, 0.756611, 0.768029),
            # (0.5276442307692307, 0.5060096153846154, 0.5709134615384616, 0.5576923076923077)
        ]

        # Scale the normalized coordinates to the actual dimensions of the image
        labels = []
        for (a_norm, b_norm, c_norm, d_norm) in normalized_labels:

            a = int(a_norm * width)
            b = int(b_norm * height)
            c = int(c_norm * width)
            d = int(d_norm * height)
            #test when we don't know the label format, can we draw the bounding box with cv2.rectangle
            # we have to transfrom the label to match the format of cv2.rectangle in each case

            # Format     u"xyxy"
            xyxy = (a, b, c, d)
            # Format     "yxyx"
            yxyx = (b, a, d, c)
            # Format     "xywh"
            xywh = (a, b, c, d)
            # Format     "center_xywh"
            center_xywh = (a - c / 2, b - d / 2, a+c/2, b+d/2)
            # Format     "rel_xyxy"
            rel_xyxy = (a / width, b / height, c / width, d / height)
            # Format     "rel_yxyx"
            rel_yxyx = (b / height, a / width, d / height, c / width)
            # Format     "rel_xywh"
            rel_xywh = (a / width, b / height, c / width, d / height)
            # Format     "rel_center_xywh"
            rel_center_xywh = (a / width + c / width / 2, b / height + d / height / 2, c / width, d / height)
            # Format     "corners_xy"
            corners_xy = [(a, b), (a + c, b), (a + c, b + d), (a, b + d)]
            # Format     "polygon"
            polygon = [(a, b), (a + c, b), (a + c, b + d), (a, b + d)]


            # Thêm các định dạng vào danh sách labels
            labels.append({
                "xyxy": xyxy,
                "yxyx": yxyx,
                "xywh": xywh,
                "center_xywh": center_xywh,
                "rel_xyxy": rel_xyxy,
                "rel_yxyx": rel_yxyx,
                "rel_xywh": rel_xywh,
                "rel_center_xywh": rel_center_xywh,
                "corners_xy": corners_xy,
                "polygon": polygon
            })

        # Vẽ bounding boxes trên hình ảnh (chỉ sử dụng định dạng "xyxy" để vẽ)
        for label in labels:
            x1, y1, x2, y2 = label["center_xywh"]
            print(x1, y1, x2, y2)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Hiển thị hình ảnh
        cv2.imshow('Image with Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
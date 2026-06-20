import cv2
import os
import glob
import numpy as np
from typing import Tuple

def extract_flower_mask(img: np.ndarray) -> np.ndarray:
    """
    Tạo mặt nạ nhị phân loại bỏ màu xanh lá cây (lá, cỏ) để giữ lại vùng hoa.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Xác định dải màu xanh lá cây trong không gian HSV của OpenCV
    lower_green = np.array([18, 40, 40])
    upper_green = np.array([43, 255, 255])
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    # Đảo ngược mặt nạ để lấy vùng không phải xanh lá cây
    flower_mask = cv2.bitwise_not(green_mask)
    return flower_mask

def preprocess_single_image(img: np.ndarray, targetedSize: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Tiền xử lý một ảnh: Resize ảnh về kích thước chỉ định và làm mờ bằng Gaussian Blur.
    
    Args:
        img (np.ndarray): Ảnh đầu vào đọc bằng OpenCV (BGR).
        targetedSize (Tuple[int, int]): Kích thước đích (width, height).
        
    Returns:
        np.ndarray: Ảnh đã tiền xử lý (BGR).
    """
    resized_img = cv2.resize(img, targetedSize, interpolation=cv2.INTER_LINEAR)
    # Tham số sigmaX = 0 để tự động tính toán độ mờ dựa trên kernel size (5x5)
    blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
    return blurred_img

def preprocess(inputFolder: str, outputFolder: str, targetedSize: Tuple[int, int] = (256, 256)) -> None:
    """
    Tiền xử lý toàn bộ các tệp ảnh trong thư mục đầu vào và lưu kết quả vào thư mục đầu ra.
    
    Args:
        inputFolder (str): Đường dẫn thư mục chứa ảnh gốc.
        outputFolder (str): Đường dẫn thư mục lưu ảnh đã làm sạch.
        targetedSize (Tuple[int, int]): Kích thước đích.
    """
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    imagePaths = glob.glob(os.path.join(inputFolder, '*.[jp][pn]g'))

    if not imagePaths:
        print("khong tim thay anh")
        return

    for img_path in imagePaths:
        filename = os.path.basename(img_path)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Gọi hàm xử lý đơn lẻ cho từng ảnh (Sửa lỗi thụt lề từ phiên bản gốc)
        blurred_img = preprocess_single_image(img, targetedSize)
        output_path = os.path.join(outputFolder, f"processed_{filename}")
        cv2.imwrite(output_path, blurred_img)
        
    print("xu ly hoan tat")

if __name__ == "__main__":
    input_folder = "./rawimg"
    output_folder = "./cleanedimg"
    preprocess(input_folder, output_folder, targetedSize=(256, 256))

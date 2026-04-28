import cv2
import os
import glob
def preprocess(inputFolder,outputFolder,targetedSize=(256,256)):
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    imagePaths=glob.glob(os.path.join(inputFolder,'*.[jp][pn]g'))

    if not imagePaths:
        print("khong tim thay anh")
        return

    for img_path in imagePaths:
        filename = os.path.basename(img_path)

        img=cv2.imread(img_path)
        if img is None:
            continue

    resized_img=cv2.resize(img,targetedSize,interpolation=cv2.INTER_LINEAR)
    # Tham số sigmaX = 0 để tự động tính toán độ mờ dựa trên kernel size
    blurred_img=cv2.GaussianBlur(resized_img,(5,5),0)

    output_path=os.path.join(outputFolder, f"processed_{filename}")
    cv2.imwrite(output_path,blurred_img)
    print("xu ly hoan tat")

if __name__ == "__main__":
    input_folder = "./rawimg"
    output_folder = "./cleanedimg"
    preprocess(input_folder,output_folder,targetedSize=(256,256))
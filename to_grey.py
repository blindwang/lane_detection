import cv2
from skimage import measure, color
from skimage.measure import regionprops
import numpy as np
import os
import copy


def skimageFilter(gray):
    binary_warped = copy.copy(gray)
    binary_warped[binary_warped > 0.1] = 255
    gray = (np.dstack((gray, gray, gray)) * 255).astype('uint8')
    labels = measure.label(gray[:, :, 0], connectivity=1)
    dst = color.label2rgb(labels, bg_label=0, bg_color=(0, 0, 0))
    gray = cv2.cvtColor(np.uint8(dst * 255), cv2.COLOR_RGB2GRAY)
    return binary_warped, gray


def moveImageTodir(path, targetPath, name):
    if os.path.isdir(path):
        image_name = os.path.join(targetPath, "image", str(name) + ".png")
        binary_name = os.path.join(targetPath, "gt_image_binary", str(name) + ".png")
        instance_name = os.path.join(targetPath, "gt_image_instance", str(name) + ".png")
        train_rows = image_name + " " + binary_name + " " + instance_name + "\n"
        origin_img = cv2.imread(os.path.join(path, "img.png"))
        origin_img = cv2.resize(origin_img, (1280, 720))
        cv2.imwrite(image_name, origin_img)
        img = cv2.imread(os.path.join(path, 'label.png'))
        img = cv2.resize(img, (1280, 720))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_warped, instance = skimageFilter(gray)
        cv2.imwrite(binary_name, binary_warped)
        cv2.imwrite(instance_name, instance)
        print("success create data name is : ", train_rows)
        return train_rows
    return None


if __name__ == "__main__":
    count = 0
    with open("train2.txt", 'w+') as file:  # 路径替换成自己的相对路径或者绝对路径
        # for images_dir in os.listdir("./images"):
        dir_name = os.path.join(r"data\source_image")  # 路径替换成自己的相对路径或者绝对路径
        for annotations_dir in os.listdir(dir_name):
            json_dir = os.path.join(dir_name, annotations_dir)
            print(json_dir)
            if os.path.isdir(json_dir):
                train_rows = moveImageTodir(json_dir, r"data\training_data_example", str(count).zfill(4))  # 路径替换
                file.write(train_rows)
                count += 1

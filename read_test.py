from PIL import Image
import numpy as np
import os

root_dir = "d:\pycharm_1/ai_test\search/train_data/train_label"

def gen_list():
    img_list = []
    for parent, dirnames, filenames in os.walk(root_dir):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for filename in filenames:  # 输出文件信息
            img_list.append(filename.replace(".jpg", ""))
            # print("parent is:" + parent)
            # print("filename is:" + filename)
            # print("the full name of the file is:" + os.path.join(parent, filename))  # 输出文件路径信息
    return img_list

img_list = gen_list()
print(img_list)
def get_test_captcha_text_and_image(i=None):
    img = img_list[i]
    captcha_image = Image.open(root_dir + "\\" + img + ".jpg")
    # captcha_image = captcha_image.resize((160, 60))
    captcha_image = np.array(captcha_image)
    return img, captcha_image

def get_test_sets_length():
    return len(img_list)
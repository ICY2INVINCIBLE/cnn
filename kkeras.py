# 导入所需工具包
from CNN_net import SimpleVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# 读取数据和标签
print("------开始读取数据------")
data = []
labels = []

# 拿到图像数据路径，方便后续读取
#imagePaths = sorted(list(paths.list_images('ai_test/search/fenlei')))
imagePaths="d:/pycharm_1/ai_test/search/fenlei"
print(imagePaths)
random.seed(42)
#random.shuffle(imagePaths)
dir_counter=0

# 遍历读取数据
for child_dir in os.listdir(imagePaths):
    #print(child_dir)

    child_path = os.path.join(imagePaths, child_dir)

    for dir_image in os.listdir(child_path):
        img = cv2.imread(os.path.join(child_path, dir_image))
        img=cv2.resize(img,(16,20))
        img = img / 255.0
        data.append(img)
       # print(dir_image.split(".")[0])
        labels.append(child_dir)
        #print(labels)
    dir_counter += 1

# 对图像数据做scale操作
#data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
data = np.array(data)
#print(labels)
#print(data)
print(len(labels))



# 数据集切分
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

# 转换标签为one-hot encoding格式
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
print(len(lb.classes_))

# 数据增强处理
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# 建立卷积神经网络
model = SimpleVGGNet.build(width=16, height=20, depth=3,classes=len(lb.classes_))

# 设置初始化超参数
INIT_LR = 0.01
EPOCHS = 50
BS = 32

# 损失函数，编译模型
print("------准备训练网络------")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# 训练网络模型
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS)
"""
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=32)
"""


# 测试
print("------测试网络------")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

# 绘制结果曲线
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('output/cnn_plot.png')

# 保存模型
print("------正在保存模型------")
model.save('output/cnn.model')
f = open('output/cnn_lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()
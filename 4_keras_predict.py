# 导入所需工具包
from tokenize import String

from keras.models import load_model
import argparse
import pickle
import cv2
import csv
model = load_model('cnn1347test60.model')
f=open('cnn1347test60.csv','w',encoding='utf-8',newline='')
csv_writer=csv.writer(f)
csv_writer.writerow(['ID','label'])
# 加载测试数据并进行相同预处理操作
def predict(model):
    for k in range(1,5001):
        list=[]
        ii = str(k) + ".jpg"
        list.append(ii)
        a=''
        for j in range(0,4):
            path="d:/pycharm_1/ai_test/search/test_split/"+str(k)+"_"+str(j)+".jpg"
            #print(path)
            image = cv2.imread(path)
         # output = image.copy()
            image = cv2.resize(image, (30, 40))

    # scale图像数据
            image = image / 255.0

    # 对图像进行拉平操作
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # 读取模型和标签
            print("------读取模型和标签------")
    #model = load_model('output/cnn.model')
            lb = pickle.loads(open('output/cnn_lb.pickle', "rb").read())

    # 预测
            preds = model.predict(image)
    # print(preds)
    # 得到预测结果以及其对应的标签
            i = preds.argmax(axis=1)[0]
            label = lb.classes_[i]
            #print(label)
            if(label=='1A'):
                label='A'
                #list.append('A')
            elif (label=='1B'):
                label = 'B'
                #list.append('B')
            elif (label=='1C'):
                label = 'C'
            elif (label=='1D'):
                label = 'D'
            elif (label=='1E'):
                label = 'E'
            elif (label=='1F'):
                label = 'F'
            elif (label=='1G'):
                label = 'G'
            elif (label=='1H'):
                label = 'H'
            elif (label=='1I'):
                label = 'I'
            elif (label=='1J'):
                label = 'J'
            elif (label=='1K'):
                label = 'K'
            elif (label=='1L'):
                label = 'L'
            elif (label=='1M'):
                label = 'M'
            elif (label=='1N'):
                label = 'N'
            elif (label=='1O'):
                label = 'O'
            elif (label=='1P'):
                label = 'P'
            elif (label=='1Q'):
                label = 'Q'
            elif (label=='1R'):
                label = 'R'
            elif (label=='1S'):
                label = 'S'
            elif (label=='1T'):
                label = 'T'
            elif (label=='1U'):
                label = 'U'
            elif (label=='1V'):
                label = 'V'
            elif (label=='1W'):
                label = 'W'
            elif (label=='1X'):
                label = 'X'
            elif (label=='1Y'):
                label = 'Y'
            elif (label=='1Z'):
                label = 'Z'

            a=a+label
        list.append(a)
        print(list)
        csv_writer.writerow(list)

        #将list加到第一行

# 在图像中把结果画出来
#text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
#cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
predict(model)
# 绘图
#cv2.imshow("Image", output)
#cv2.waitKey(0)
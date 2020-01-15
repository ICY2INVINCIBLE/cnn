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

imagePaths = sorted(list(paths.list_images('./dataset')))
random.seed(42)
random.shuffle(imagePaths)

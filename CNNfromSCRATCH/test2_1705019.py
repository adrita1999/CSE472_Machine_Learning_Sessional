#!/usr/bin/env python
# coding: utf-8

# # Imports



import sklearn as sklearn
# import keras.datasets as datasets
# from keras.datasets import cifar10
from PIL import Image
import numpy as np
import os
import pandas as pd
import math
import cv2
import random
import csv
from sklearn.metrics import f1_score
import pickle as pkl
from numpy.lib.stride_tricks import as_strided
from math import floor
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# np.random.seed(1)


# # Load Data
#



# path to the folders where the images are stored
path_training_folder = 'E:/4--2/CSE 472 (ML Lab)/offline 4/NumtaDB_with_aug/training-a'
# path_training_folder='training-n'
training_csv= 'E:/4--2/CSE 472 (ML Lab)/offline 4/NumtaDB_with_aug/training-a.csv'

path_testing_folder = 'E:/4--2/CSE 472 (ML Lab)/offline 4/NumtaDB_with_aug/training-d'
testing_csv= 'E:/4--2/CSE 472 (ML Lab)/offline 4/NumtaDB_with_aug/training-d.csv'

path_validating_folder = 'E:/4--2/CSE 472 (ML Lab)/offline 4/NumtaDB_with_aug/training-c'
validating_csv= 'E:/4--2/CSE 472 (ML Lab)/offline 4/NumtaDB_with_aug/training-c.csv'

L_RATE = 0.001

def load_test(given_path):
    x_test = []
    image_name = []

    # get the list of all images in the folder
    train_img_list = [f for f in os.listdir(given_path) if f.endswith('.jpg') or f.endswith('.png')]

    # convert each image into a matrix
    for img_name in train_img_list:
        img = Image.open(os.path.join(given_path, img_name))
        # print(img_name)
        img = img.resize((28, 28))
        img_matrix = np.array(img)
        img_matrix = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2RGB)
        # print("Image Matrix Shape:", img_matrix.shape)
        x_test.append(img_matrix)
        image_name.append(img_name)

    x_test = np.array(x_test)
    x_test = x_test / 255
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    # print("Training Image Matrix Shape:", x_train.shape)
    # image_info = []
    # with open(testing_csv) as csvfile:
    #     # print("opened")
    #     reader = csv.reader(csvfile)
    #     next(reader) # skip header row
    #     for row in reader:
    #         # print(row)
    #         image_name = row[6]
    #         print(image_name)
    #         image_path= os.path.join(path_testing_folder, image_name)
    #         label = int(row[2])
    #         print(label)
    #         image_info.append((image_path, label))
    return x_test, np.array(image_name)


# # Convolution Layer

class convolutionLayer:
    def __init__(self,in_c, out_c, kernel_size=3, stride=1, padding=0, learn_rate=L_RATE):
        self.nb_input_channels = in_c
        self.nb_output_channels = out_c
        # self.bias = None
        # self.weight = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.learning_rate = learn_rate

        self.cache = None

        # self.weight = np.random.randn(self.nb_output_channels, self.nb_input_channels,  self.kernel_size, self.kernel_size) * 1e-3

        self.weight = np.random.randn(self.nb_output_channels, self.nb_input_channels,  self.kernel_size, self.kernel_size)* np.sqrt(2.0 / (self.nb_input_channels * self.kernel_size * self.kernel_size))
        self.bias = np.zeros(self.nb_output_channels)


    def get_windows(self,k_size, in_matrix, out_shape, padding=0, stride=1, ifdilate=0):
        input_matrix = in_matrix
        padding_temp = padding

        # dilate the input if necessary
        if ifdilate != 0:
            input_matrix = np.insert(input_matrix, range(1, in_matrix.shape[2]), 0, axis=2)
            input_matrix = np.insert(input_matrix, range(1, in_matrix.shape[3]), 0, axis=3)

        # pad the input if necessary
        if padding_temp != 0:
            input_matrix = np.pad(input_matrix, pad_width=((0,), (0,), (padding_temp,), (padding_temp,)), mode='constant', constant_values=(0.,))

        out_bsize, out_channel, temph, tempw = in_matrix.shape
        temp_bsize, temp_channel, out_height, out_width = out_shape

        b_stride, c_stride, h_stride, w_stride = input_matrix.strides

        return np.lib.stride_tricks.as_strided(
            input_matrix,
            (out_bsize, out_channel, out_height, out_width, k_size, k_size),
            (b_stride, c_stride, stride * h_stride, stride * w_stride, h_stride, w_stride)
        )

    def forward(self, input_matrix):
        # print("weights in conv:", self.weight)
        n, c, h, w = input_matrix.shape

        ht = (h - self.kernel_size + (2 * self.padding)) // self.stride + 1
        wd = (w - self.kernel_size + (2 * self.padding)) // self.stride + 1

        windows = self.get_windows(self.kernel_size,input_matrix, (n, c, ht, wd), self.padding, self.stride)

        output_matrix = np.einsum('bihwkl,oikl->bohw', windows, self.weight)

        # add bias to kernels
        output_matrix += self.bias[None, :, None, None]

        self.cache = windows, input_matrix

        return output_matrix


    def backward(self, grad):
        windows,tempx = self.cache

        # padding = self.kernel_size - 1 if self.padding == 0 else self.padding
        if self.padding == 0:
            pad = self.kernel_size - 1
        else:
            pad = self.padding

        grad_windows = self.get_windows(self.kernel_size,grad, grad.shape, padding=pad, stride=1, ifdilate=self.stride - 1)
        kernel_rotate = np.rot90(self.weight, 2, axes=(2, 3))

        # update weights and bias
        dw = np.einsum('bihwkl,bohw->oikl', windows, grad)
        db = np.sum(grad, axis=(0, 2, 3))

        self.weight = self.weight - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db

        dx = np.einsum('bohwkl,oikl->bihw', grad_windows, kernel_rotate)
        dout= dx

        return dout

    def reset(self):
        self.cache = None


# # ReLU Layer

class ReLU:
    def __init__(self):
        self.matrix = None

    def forward(self, matrix):
        self.matrix = matrix
        return np.maximum(0, matrix)

    def backward(self, out):
        temp = np.array(out, copy=True)
        temp[self.matrix <= 0] = 0
        return temp
        # return grad * (self.x > 0)

    def reset(self):
        self.matrix = None


# # Max Pooling Layer


class maxpoolLayer:
    def __init__(self, k_size, stride, padding=0):
        self.kernel_size = k_size
        self.stride = stride
        self.padding = padding

        self.cache = {}
        self.pool_size = (k_size,k_size)


    def forward(self,input_matrix):
        n_batch, ch_x, h_x, w_x = input_matrix.shape
        h_poolwindow,w_poolwindow=self.pool_size
        # w_poolwindow = self.pool_size

        out_h = int((h_x - h_poolwindow) / self.stride) + 1
        out_w = int((w_x - w_poolwindow) / self.stride) + 1
        windows = as_strided(input_matrix,
                     shape=(n_batch, ch_x, out_h, out_w, *self.pool_size),
                     strides=(input_matrix.strides[0], input_matrix.strides[1],
                              self.stride * input_matrix.strides[2],
                              self.stride * input_matrix.strides[3],
                              input_matrix.strides[2], input_matrix.strides[3])
                     )

        out = np.max(windows, axis=(4, 5))

        maxs = out.repeat(2, axis=2).repeat(2, axis=3)
        x_window = input_matrix[:, :, :out_h * self.stride, :out_w * self.stride]
        mask = np.equal(x_window, maxs).astype(int)

        self.cache['IN'] = input_matrix
        self.cache['mask'] = mask
        # print("cache : ",self.cache)

        return out

    def backward(self, grad):
        h_pool, w_pool = self.pool_size
        forward_matrix = self.cache['IN']
        mask = self.cache['mask']

        gradA = grad.repeat(h_pool, axis=2).repeat(w_pool, axis=3)
        gradA = np.multiply(gradA, mask)

        pad = np.zeros(forward_matrix.shape)
        pad[:, :, :gradA.shape[2], :gradA.shape[3]] = gradA
        return pad

    def reset(self):
        self.cache = {}


# # Flattening Layer


class flattenLayer:
    def __init__(self):
        self.matrix_shape = None

    def forward(self, matrix):
        # print("matrix shape")
        # print(matrix.shape)
        self.matrix_shape=matrix.shape
        flatenned_result = np.copy(matrix)
        no_samples=flatenned_result.shape[0]
        len_flatenned_result = np.prod(flatenned_result.shape[1:])
        flatenned_result = np.reshape(flatenned_result, (no_samples, len_flatenned_result))
        # flatenned_result = flatenned_result.transpose()
        return flatenned_result

    def backward(self, out):
        # out=out.transpose()
        return out.reshape(self.matrix_shape)

    def reset(self):
        self.matrix_shape = None


# # Fully Connected Layer



class fullyConnectedLayer:
    def __init__(self, in_size, out_size,learning_rate):
        self.in_size = in_size
        self.out_size = out_size
        factor=np.sqrt(2/self.in_size)
        self.weights = np.random.randn(self.out_size, self.in_size) * factor
        self.bias = np.zeros((self.out_size, 1))
        self.learning_rate = L_RATE
        self.forward_in_matrix = None

    def forward(self, in_matrix):
        self.forward_in_matrix = in_matrix
        x = np.dot(self.weights, in_matrix.T)
        x = x + self.bias
        return x.T

    def backward(self, gradient):
        grad_X=np.zeros(self.forward_in_matrix.shape)
        grad_X=np.dot(gradient,self.weights)

        grad_W=np.zeros(self.weights.shape)
        grad_W=np.dot(gradient.T,self.forward_in_matrix)

        grad_B=np.zeros(self.bias.shape)
        grad_B=np.sum(gradient.T,axis=1,keepdims=True)

        factor_w=(grad_W*self.learning_rate)
        self.weights = self.weights - factor_w
        factor_b=(grad_B*self.learning_rate)
        self.bias = self.bias - factor_b

        return grad_X.reshape(self.forward_in_matrix.shape)

    def reset(self):
        self.forward_in_matrix= None
        self.in_size = None
        self.out_size = None
        self.learning_rate = None


# # Softmax Layer



class softmaxLayer:
    def __init__(self):
        pass
    def forward(self, forward_in_matrix):
        temp_exp = np.exp(forward_in_matrix)
        result = temp_exp / np.sum(temp_exp, axis=1, keepdims=True)
        return result

    def backward(self, dout):
        return np.copy(dout)
    def reset(self):
        pass


# # Model



class myModel:
    def __init__(self,dim,channel_count):
        self.conv1_layer=convolutionLayer(3,6,5,1,2)
        dim=floor((dim-5+(2*2))/1)+1
        self.relu1_layer=ReLU()
        self.pool1_layer=maxpoolLayer(2,2)
        dim=floor((dim-2)/2)+1
        self.conv2_layer=convolutionLayer(6,16,5,1,2)
        channel_count=16
        dim=floor((dim-5+(2*2))/1)+1
        # dim minus kernel plus (2*padding)/stride +1
        self.relu2_layer=ReLU()
        self.pool2_layer=maxpoolLayer(2,2)
        dim=floor((dim-2)/2)+1
        input1=int(dim*dim*channel_count)
        # print("input1 : ",input1)
        self.flat_layer=flattenLayer()
        self.fc1_layer=fullyConnectedLayer(input1,120,L_RATE)
        self.relu3_layer=ReLU()
        self.fc2_layer=fullyConnectedLayer(120,84,L_RATE)
        self.relu4_layer=ReLU()
        self.fc3_layer=fullyConnectedLayer(84,10,L_RATE)
        self.softmax_layer=softmaxLayer()
    def forward(self,matrix):
        matrix=self.conv1_layer.forward(matrix)
        # print("forward conv1_layer output shape : ",matrix.shape)
        matrix=self.relu1_layer.forward(matrix)
        # print("forward relu1_layer output shape : ",matrix.shape)
        matrix=self.pool1_layer.forward(matrix)
        # print("forward pool1_layer output shape : ",matrix.shape)
        matrix=self.conv2_layer.forward(matrix)
        # print("forward conv2_layer output shape : ",matrix.shape)
        matrix=self.relu2_layer.forward(matrix)
        # print("forward relu2_layer output shape : ",matrix.shape)
        matrix=self.pool2_layer.forward(matrix)
        # print("forward pool2_layer output shape : ",matrix.shape)
        matrix=self.flat_layer.forward(matrix)
        # print("forward flat_layerten output shape : ",matrix.shape)
        matrix=self.fc1_layer.forward(matrix)
        # print("forward fc1_layer output shape : ",matrix.shape)
        matrix=self.relu3_layer.forward(matrix)
        matrix=self.fc2_layer.forward(matrix)
        # print("forward fc2_layer output shape : ",matrix.shape)
        matrix=self.relu4_layer.forward(matrix)
        matrix=self.fc3_layer.forward(matrix)
        # print("forward fc3_layer output shape : ",matrix.shape)
        matrix=self.softmax_layer.forward(matrix)
        # print("softmax_layer:", matrix)
        row_sums = np.sum(matrix, axis=1)
        # print("row_sums:", row_sums)
        return matrix
    def backward(self,out_matrix):
        out_matrix=self.softmax_layer.backward(out_matrix)
        # print("backward softmax_layer output shape : ",out_matrix.shape)
        out_matrix=self.fc3_layer.backward(out_matrix)
        # print("backward fc3_layer output shape : ",out_matrix.shape)
        out_matrix=self.relu4_layer.backward(out_matrix)
        out_matrix=self.fc2_layer.backward(out_matrix)
        # print("backward fc2_layer output shape : ",out_matrix.shape)
        out_matrix=self.relu3_layer.backward(out_matrix)
        out_matrix=self.fc1_layer.backward(out_matrix)
        # print("backward fc1_layer output shape : ",out_matrix.shape)
        out_matrix=self.flat_layer.backward(out_matrix)
        # print("backward flat_layerten output shape : ",out_matrix.shape)
        out_matrix=self.pool2_layer.backward(out_matrix)
        # print("backward pool2_layer output shape : ",out_matrix.shape)
        out_matrix=self.relu2_layer.backward(out_matrix)
        # print("backward relu2_layer output shape : ",out_matrix.shape)
        out_matrix=self.conv2_layer.backward(out_matrix)
        # print("backward conv2_layer output shape : ",out_matrix.shape)
        out_matrix=self.pool1_layer.backward(out_matrix)
        # print("backward pool1_layer output shape : ",out_matrix.shape)
        out_matrix=self.relu1_layer.backward(out_matrix)
        # print("backward relu1_layer output shape : ",out_matrix.shape)
        out_matrix=self.conv1_layer.backward(out_matrix)
        # print("backward conv1_layer output shape : ",out_matrix.shape)
        return out_matrix
    def reset(self):
        self.conv1_layer.reset()
        self.conv2_layer.reset()
        self.fc1_layer.reset()
        self.fc2_layer.reset()
        self.fc3_layer.reset()
        self.softmax_layer.reset()
        self.relu1_layer.reset()
        self.relu2_layer.reset()
        self.relu3_layer.reset()
        self.relu4_layer.reset()
        self.pool1_layer.reset()
        self.pool2_layer.reset()
        self.flat_layer.reset()


# # Metrics



def crossEntropyLoss(y_true, y_pred):
    # y_true = np.clip(y_true, 1e-7, 1-1e-7)
    # y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    # loss = -np.sum(y_true * np.log(y_pred))
    return np.sum(-np.log(y_pred) * y_true)

def f1_score_metric(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1

def accuracy_metric(y_true,y_pred):
    return sklearn.metrics.accuracy_score(y_true,y_pred)


# # Helper Functions



def split_list(input_list, split_ratio):
    split_index = int(len(input_list) * split_ratio)
    return input_list[:split_index], input_list[split_index:]


with open('1705019_model.pkl', 'rb') as f:
    saved_model = pkl.load(f)
#test
given_path=str(sys.argv[1])
# print(given_path)

x_test,test_names=load_test(given_path)
# x_test,y_test,test_names=load_test(path_testing_folder)
# print("x_test shape: ",x_test.shape)
x_testing=np.transpose(x_test, (0, 3, 1, 2))
mout=saved_model.forward(x_testing)
y_pred=mout

predicted_label=np.argmax(y_pred,axis=1)

with open('1705019_prediction.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["FileName","Digit"])
    for r in range(len(test_names)):
        writer.writerow([test_names[r],predicted_label[r]])






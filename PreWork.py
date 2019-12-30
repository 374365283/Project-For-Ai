import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import *

airplane = []
label_airplane = []
automobile = []
label_automobile = []
bird = []
label_bird = []
cat = []
label_cat = []
deer = []
label_deer = []
dog = []
label_dog = []
frog = []
label_frog = []
horse = []
label_horse = []
ship = []
label_ship = []
truck = []
label_truck = []

def get_file(file_dir):
    # step1：获取路径下所有的图片路径名，存放到
    # 对应的列表中，同时贴上标签，存放到label列表中。
    for file in os.listdir(file_dir + '/airplane'):
        airplane.append(file_dir + '/airplane' + '/' + file)
        label_airplane.append(0)
    for file in os.listdir(file_dir + '/automobile'):
        automobile.append(file_dir + '/automobile' + '/' + file)
        label_automobile.append(1)
    for file in os.listdir(file_dir + '/bird'):
        bird.append(file_dir + '/bird' + '/' + file)
        label_bird.append(2)
    for file in os.listdir(file_dir + '/cat'):
        cat.append(file_dir + '/cat' + '/' + file)
        label_cat.append(3)
    for file in os.listdir(file_dir + '/deer'):
        deer.append(file_dir + '/deer' + '/' + file)
        label_deer.append(4)
    for file in os.listdir(file_dir + '/dog'):
        dog.append(file_dir + '/dog' + '/' + file)
        label_dog.append(5)
    for file in os.listdir(file_dir + '/frog'):
        frog.append(file_dir + '/frog' + '/' + file)
        label_frog.append(6)
    for file in os.listdir(file_dir + '/horse'):
        horse.append(file_dir + '/horse' + '/' + file)
        label_horse.append(7)
    for file in os.listdir(file_dir + '/ship'):
        ship.append(file_dir + '/ship' + '/' + file)
        label_ship.append(8)
    for file in os.listdir(file_dir + '/truck'):
        truck.append(file_dir + '/truck' + '/' + file)
        label_truck.append(9)
    # step2：对生成的图片路径和标签List做打乱处理把所有的合起来组成一个list（img和lab）
    # 合并数据numpy.hstack(tup)
    # tup可以是python中的元组（tuple）、列表（list），或者numpy中数组（array），函数作用是将tup在水平方向上（按列顺序）合并
    image_list = np.hstack((airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck))
    label_list = np.hstack((label_airplane, label_automobile, label_bird, label_cat, label_deer, label_dog, label_frog, label_horse, label_ship, label_truck))
    # 利用shuffle，转置、随机打乱
    temp = np.array([image_list, label_list])   # 转换成2维矩阵
    temp = temp.transpose()     # 转置
    # numpy.transpose(a, axes=None) 作用：将输入的array转置，并返回转置后的array
    np.random.shuffle(temp)     # 按行随机打乱顺序函数

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])    # 取出第0列数据，即图片路径
    all_label_list = list(temp[:, 1])    # 取出第1列数据，即图片标签
    label_list = [int(i) for i in label_list]   # 转换成int数据类型

    return image_list, label_list


# image_W, image_H, ：图像高度和宽度
# batch_size：每个batch要放多少张图片
# capacity：一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue
    # tf.cast()用来做类型转换
    image = tf.cast(image, tf.string)   # 可变长度的字节数组.每一个张量元素都是一个字节数组
    label = tf.cast(label, tf.int32)
    # tf.train.slice_input_producer是一个tensor生成器
    # 作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])   # tf.read_file()从队列中读取图像

    # step2：将图像解码，使用相同类型的图像
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 对resize后的图片进行标准化处理
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)

    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])

    image_batch = tf.cast(image_batch, tf.float32)    # 显示灰度图
    # print(label_batch) Tensor("Reshape:0", shape=(6,), dtype=int32)
    return image_batch, label_batch
    # 获取两个batch，两个batch即为传入神经网络的数据



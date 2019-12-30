import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from NeuralNet import net1, net2, net3

N_CLASSES = 10

img_dir = 'E:/autumn 2019/brain-like intelligence/CNN/test_dataset/'
log_dir = 'E:/autumn 2019/brain-like intelligence/CNN/train_logs/200_2'
lists = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def get_one_image(img_dir):
    imgs = os.listdir(img_dir)
    img_num = len(imgs)
    # print(imgs, img_num)
    idn = np.random.randint(0, img_num)
    image = imgs[idn]
    image_dir = img_dir + image
    print(image_dir)
    print(image[0])
    image = Image.open(image_dir)
    #plt.imshow(image)
    #plt.show()
    image = image.resize([32, 32])
    image_arr = np.array(image)
    return image_arr


def get_all_image(image_dir):
    imgs = os.listdir(img_dir)
    img_num = len(imgs)
    # print(imgs, img_num)
    image_array = []
    image_category = []
    for idn in range(img_num):
        image = imgs[idn]
        image_dir = img_dir + image
        #print(image_dir)
        image = Image.open(image_dir)
        # plt.imshow(image)
        # plt.show()
        image = image.resize([32, 32])
        image_arr = np.array(image)
        image_array.append(image_arr)
        image_category.append(ord(image_dir[56]) - ord('0'))


    return image_array , image_category

def test(image_array , image_category):
    correct_num = 0
    img_num = len(image_array)
    for index in range(img_num):
        image_arr = image_array[index]
        correct_index = image_category[index]
        with tf.Graph().as_default():
            image = tf.cast(image_arr, tf.float32)
            # print('1', np.array(image).shape)
            image = tf.image.per_image_standardization(image)
            # print('2', np.array(image).shape)
            image = tf.reshape(image, [1, 32, 32, 3])
            # print(image.shape)
            p = net2(image, 1, N_CLASSES)
            logits = tf.nn.softmax(p)
            x = tf.placeholder(tf.float32, shape=[32, 32, 3])
            saver = tf.train.Saver()
            sess = tf.Session()
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 调用saver.restore()函数，加载训练好的网络模型
            # prediction = sess.run(logits, feed_dict={x: image_arr})
            prediction = sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction)
            #print('预测的标签为：', max_index, lists[max_index])
            #print('预测的结果为：', prediction)
            if(max_index == correct_index):
                correct_num += 1
    print('准确率为', correct_num/img_num)

if __name__ == '__main__':
    img, category = get_all_image(img_dir)  # 通过改变参数train or val，进而验证训练集或测试集
    test(img, category)

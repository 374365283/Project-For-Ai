import os
import xlwt
import numpy as np
import tensorflow as tf
from PreWork import get_file, get_batch
from Function import losses, trainning, evaluation
from NeuralNet import net1, net2, net3

# 变量声明
N_CLASSES = 10
IMG_W = 32  # resize图像，太大的话训练时间久
IMG_H = 32
BATCH_SIZE = 50    # 每个batch要放多少张图片
CAPACITY = 1000     # 一个队列最大多少
MAX_STEP = 50000 # 一般大于10K
learning_rate = 0.0001  # 一般小于0.0001

# 获取批次batch
document_dir = 'E:/autumn 2019/brain-like intelligence/CNN'
train_dir = 'E:/autumn 2019/brain-like intelligence/CNN/train_dataset_1000'  # 训练样本的读入路径
logs_train_dir = 'E:/autumn 2019/brain-like intelligence/CNN/train_logs/1000_1'  #logs存储路径
train, train_label = get_file(train_dir)
# 训练数据及标签
train_batch, train_label_batch = get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# 训练操作定义
train_logits = net1(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = losses(train_logits, train_label_batch)
train_op = trainning(train_loss, learning_rate)
train_acc = evaluation(train_logits, train_label_batch)

# 这个是log汇总记录
summary_op = tf.summary.merge_all()

# 产生一个会话
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
# 产生一个saver来存储训练好的模型
saver = tf.train.Saver()
# 所有节点初始化
sess.run(tf.global_variables_initializer())
# 队列监控
coord = tf.train.Coordinator() # 设置多线程协调器
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 进行batch的训练
try:
    # 执行MAX_STEP步的训练，一步一个batch
    path = logs_train_dir+'/train_net1.xlsx'
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet('Sheet1')
    sheet_row = 1
    train_loss_array = []
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

        # 每隔100步打印一次当前的loss以及acc，同时记录log，写入writer
        if step % 100 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            train_loss_array.append(tra_loss)
            sheet1.write(sheet_row , 0, str(step))
            sheet1.write(sheet_row , 1, str(tra_loss))
            sheet1.write(sheet_row , 2, str(tra_acc * 100.0))
            sheet_row += 1

            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)


    train_loss_gradient = np.gradient(train_loss_array)
    for i in range(len(train_loss_gradient)):
        sheet1.write( i+1 , 3, str(train_loss_gradient[i]))

    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path)
    workbook.save(path)


except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')


finally:
    coord.request_stop()
coord.join(threads)
sess.close()


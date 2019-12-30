"""
# 功能：观察一次函数采样数据集大小对误差误差的影响
# 设定不同的误差阈值，达到阈值即结束程序。最后绘制误差与需要达到迭代次数的关系
# 人为设定初始权重以及偏差，即所有迭代起始权重都相同（控制变量）
# 采样范围：【-10，10】
# 2019/12/01
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import sys

# 设定结束阈值numpy数组
ERROR = np.arange(16, 0.2, -0.1)
# 采样规模
SAMPLESIZE = [10, 20, 40, 80]
# 建立模型


class MyRegression(torch.nn.Module):
    def __init__(self):
        super(MyRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        '''
        #这里在初始化时候给权值和偏置赋定值，为了控制变量，不然初始值不同最后
        #得到的误差也不一样。当然系统会自动赋值（如果认为不设定）
        '''
        self.linear.weight = torch.nn.Parameter(torch.Tensor(np.full((1, 1), -0.5)))
        self.linear.bias = torch.nn.Parameter(torch.Tensor(np.full((1, 1), -0.5)))

    def forward(self, x):
        out = self.linear(x)
        return out


def myfunction(x, y):
    # 定义训练模型，如果支持cuda加速则采用
    if torch.cuda.is_available():
        model = MyRegression().cuda()
        # 将x,y转换成为变量
        x = Variable(torch.Tensor(x.reshape(len(x), 1))).cuda()
        y = Variable(torch.Tensor(y.reshape(len(y), 1))).cuda()
        print("布置在cuda上运算！")
    else:
        print("cuda不可用！")
        sys.exit()

    # 定义优化方法及学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    # 定义损失函数
    loss_function = torch.nn.MSELoss()
    # 迭代过程
    count_times = 0
    j = 0                   # 定位误差值
    iteration_times = []    # 迭代次数清零
    while(True):
        count_times += 1
        prediction = model(x)
        loss = loss_function(prediction, y)
        if loss.cpu().detach().numpy() <= ERROR[j]:
            j += 1
            iteration_times.append(count_times)
            # 程序出口，返回迭代每次的迭代次数
            if j == len(ERROR):
                return iteration_times
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 绘制结果图层
plt.figure("loss & iteration_times")
plt.subplot(111)
plt.title('loss')
plt.ylabel('iteration_times')
plt.xlabel('loss')
# 构造绘图数据
for i in range(len(SAMPLESIZE)):
    x = np.arange(-10, 10, 20/SAMPLESIZE[i])
    y = 4*x+2
    plt.plot(ERROR, myfunction(x, y), label='Sample size: '+str(SAMPLESIZE[i]))
    # 图例位置自动设置
    plt.legend(loc=0)
plt.show()


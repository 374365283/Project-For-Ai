"""
# 功能：观察一次函数采样数据集大小对误差误差的影响
# 固定了迭代次数
# 人为设定初始权重以及偏差，即所有迭代起始权重都相同（控制变量）
# 采样范围：【-10，10】
# 2019/12/01
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import sys

# 确定训练次数
EPCOH = 200
# 采样规模
SAMPLESIZE = [20, 40, 60, 80, 100]
# 建立模型


class MyRegression(torch.nn.Module):
    def __init__(self):
        super(MyRegression, self).__init__()
        self.linear = torch.nn.Linear(3, 1)
        '''
        #这里在初始化时候给权值和偏置赋定值，为了控制变量，不然初始值不同最后
        #得到的误差也不一样。当然系统会自动赋值（如果认为不设定）
        '''
        self.linear.weight = torch.nn.Parameter(torch.Tensor(np.full((1, 3), -0.5)))
        self.linear.bias = torch.nn.Parameter(torch.Tensor(np.full((1, 1), -0.5)))

    def forward(self, x):
        out = self.linear(x)
        return out


def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)


def myfunction(x, y):
    # 定义训练模型，如果支持cuda加速则采用
    if torch.cuda.is_available():
        model = MyRegression().cuda()
        # 将x,y转换成为变量

        x = Variable(x).cuda()
        y = Variable(torch.Tensor(y.reshape(len(y), 1))).cuda()
        print("布置在cuda上运算！")
    else:
        print("cuda不可用！")
        sys.exit()

    # 定义优化方法及学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    # 定义损失函数
    loss_function = torch.nn.MSELoss()
    lossValue = []
    print("Initial value of model")
    # for paramenters in model.parameters():
    #     print(paramenters)
    # # 迭代过程
    for i in range(EPCOH):
        prediction = model(x)
        loss = loss_function(prediction, y)
        # if i % 20 == 0:
        # #每20次打印出一次误差
        print("loss is :", loss.cpu().detach().numpy())
        # 存储误差，后期比较！
        lossValue.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 返回每次的误差
    print("迭代完一轮！")
    return lossValue


# 绘制结果图层
plt.figure("Sample size & loss")
plt.subplot(111)
plt.title('loss')
plt.xlabel('epoch times')
plt.ylabel('loss per time')
for i in range(len(SAMPLESIZE)):
    x = np.arange(-10, 10, 20/SAMPLESIZE[i])
    y = 3*x**3+2*x**2-9*x+2
    x = make_features(torch.Tensor(x))
    lossValue = myfunction(x, y)
    # 显示图例
    plt.plot(np.arange(EPCOH), lossValue, label='Sample size: '+str(SAMPLESIZE[i]))
    # 图例位置自动设置
    plt.legend(loc=0)
# 构造绘图数据

plt.show()



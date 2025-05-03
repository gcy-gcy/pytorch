import os
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import Model
from torch.utils.tensorboard import SummaryWriter
import time

# ------------------1. 一些定义---------------- #
# 定义训练的设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

# 指定pth存储文件夹
pth_dir = './model_pth'
if not os.path.exists(pth_dir):  # os模块判断并创建
    os.mkdir(pth_dir)

# 训练的轮数
epoch = 10
train_step = 0
test_step = 0
lr = 1e-2

# ------------------2.构建数据集----------------- #
trans = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])

train_data = torchvision.datasets.CIFAR10('../dataset', train=True, transform=trans, download=True)
test_data = torchvision.datasets.CIFAR10('../dataset', train=False, transform=trans, download=True)

# 数据集的长度
len_train = len(train_data)
len_test = len(test_data)

# -----------------3. 加载数据集（按照batchsize=64打包）-------------- #
train_load = DataLoader(train_data, 64, shuffle=True)
test_load = DataLoader(test_data, 64, shuffle=True)

# -----------------4. 模型、损失函数、优化器、摘要器------------- #
# 构建模型
model = Model()
model.to(device)  # 用gpu训练

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
optim = torch.optim.SGD(model.parameters(), lr=lr)
# 构建tensoroboard摘要器
writer = SummaryWriter('logs_train')

# 开始训练
for i in range(epoch):
    print('-----------第 {} 轮训练开始-------------'.format(i + 1))

    # 训练步骤开始
    model.train()
    for data in train_load:  # train_load，每个循环包含了64张
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        out = model(imgs)
        loss = loss_fn(out, targets)

        # optim 优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 损失反向传播
        optim.step()  # 优化

        # writer
        if train_step % 200 == 0:
            print('训练步数: {}, Loss: {}'.format(train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), train_step)

        train_step += 1

    # 验证步骤开始
    model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():  # 没有梯度，不会对其进行调优
        for data in test_load:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            out = model(imgs)
            loss = loss_fn(out, targets)
            total_test_loss += loss.item()
            accuracy = (out.argmax(1) == targets).sum().item()
            total_test_accuracy += accuracy

    print('整体测试集上的Loss: {}'.format(total_test_loss))
    print('整体数据集上的准确率Acc: {}'.format(total_test_accuracy / len_test))
    writer.add_scalar('test_loss', total_test_loss, test_step)
    writer.add_scalar('test_Acc', total_test_accuracy / len_test, test_step)
    test_step += 1

    # 保存方式1
    torch.save(model, pth_dir + '/model_{}.pth'.format(i))
    # 保存方式2（官方推荐）
    # torch.save(model.state_dict(), pth_dir + '/model_{}.pth'.format(i))
    print('model_{}.pth 已保存'.format(i))

writer.close()

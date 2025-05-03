import os
import torch
import torchvision
from PIL import Image
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Resize成符合模型的输入大小
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

imagetype = ['bmp', 'dib', 'png', 'jpg', 'jpeg', 'pbm', 'pgm', 'ppm', 'tif', 'tiff']
imagelist_path = 'images'
imagelist = os.listdir(imagelist_path)

for imagename in imagelist:
    start = time.perf_counter()
    if imagename.split('.')[1] not in imagetype:
        print('{} is not an image.'.format(imagename))
    else:
        # ----------------读取图像---------------- #
        image_path = os.path.join(imagelist_path, imagename)
        image = Image.open(image_path)

        # ----------------调整图像---------------- #
        image = image.convert('RGB')  # 1.转为3通道图像
        image = transform(image)  # 2.调整图像尺寸为model输入的32x32
        image = torch.unsqueeze(image, 0)  # 3.升维为4维张量：[batchsize, C, H,W]
        image = image.to(device)  # 4.因为模型使用gpu训练的，所以验证时报错，让我也用gpu验证

        # ----------------加载模型-------------- #
        model = torch.load('./model_pth/model_9.pth')
        model.to(device)  # 5.model也用gpu加载，好像要比cpu快些

        # ----------------开始测试-------------- #
        model.eval()
        with torch.no_grad():
            output = model(image)  # 输出的是各类别得分

        # ----------------打印类别-------------- #
        index = output.argmax(1).item()
        print('这张图象的类别是：{}'.format(classes[index]))

        end = time.perf_counter()
        print('这张图像测试用时：{} s'.format(end - start))

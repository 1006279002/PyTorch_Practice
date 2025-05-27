"""
创建一个基础的CNN框架神经网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 计时，用于记录训练和测试时间
import time

_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # 使用MPS设备，如果不可用则使用CPU

class SimpleCNN(nn.Module):
    """
    一个简单的卷积神经网络模型，包含两个卷积层，采用残差连接和ReLU激活函数，三个全连接层，做到minist数据集的分类。
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 编码器encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 输入通道为3，输出通道为16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输入通道为16，输出通道为32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        )
        # 残差连接
        self.residual_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        # 解码器decoder
        self.decoder = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),  # 输入特征图展平后，输入大小为32*7*7
            nn.ReLU(),
            nn.Linear(128, 64),  # 全连接层
            nn.ReLU(),
            nn.Linear(64, 10)  # 输出层，10个类别
        )

    def forward(self, x):
        """
        前向传播函数，定义了处理方式
        Args:
            x: 输入数据，形状为(batch_size, 1, 28, 28)。

        Returns:
            x: 输出数据，形状为(batch_size, 10)，表示10个类别的预测分数。
        """
        residual = self.residual_conv(x) # 残差连接
        x = self.encoder(x)
        x = x + residual # 添加残差连接
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    """
    训练模型的函数，并通过hook查看中间特征图
    Args:
        model: 神经网络模型。
        device: 设备类型（CPU或GPU）。
        train_loader: 训练数据加载器。
        optimizer: 优化器。
        epoch: 当前训练的轮数。
    """
    feature_maps = []

    def hook_fn(module, input, output):
        # 只保存第一个batch的特征图
        if len(feature_maps) == 0:
            feature_maps.append(output.detach().cpu())

    # 注册hook到第一个卷积层
    hook_handle = model.encoder[0].register_forward_hook(hook_fn)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # 只可视化第一个batch的特征图
        if batch_idx == 0 and feature_maps:
            fmap = feature_maps[0][0]  # 取第一个样本的特征图
            num_channels = fmap.shape[0]
            plt.figure(figsize=(15, 8))
            for i in range(min(num_channels, 8)):
                plt.subplot(2, 4, i + 1)
                plt.imshow(fmap[i], cmap='gray')
                plt.title(f'Channel {i}')
                plt.axis('off')
            plt.suptitle('Feature Maps from Conv1')
            plt.show()
            feature_maps.clear()  # 只显示一次

    hook_handle.remove()

def test(model, device, test_loader):
    """
    测试模型的函数
    Args:
        model: 神经网络模型。
        device: 设备类型（CPU或GPU）。
        test_loader: 测试数据加载器。

    Returns:
        float: 测试集上的平均损失。
        float: 测试集上的准确率。
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 累加损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计正确预测的数量

    test_loss /= len(test_loader.dataset)  # 平均损失
    accuracy = correct / len(test_loader.dataset)  # 准确率

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * accuracy:.2f}%)\n')

    return test_loss, accuracy

def main():
    """
    主函数，设置数据加载器，模型，优化器等，并进行训练和测试。
    """
    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 初始化模型，优化器
    model = SimpleCNN().to(_device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和测试
    for epoch in range(1, 11):
        # 记录训练开始时间
        start_time = time.time()
        train(model, _device, train_loader, optimizer, epoch)
        # 记录训练结束时间
        end_time = time.time()
        print(f"Epoch {epoch} training time: {end_time - start_time:.2f} seconds")
        test(model, _device, test_loader)

if __name__ == '__main__':
    """
    入口函数，调用主函数进行训练和测试。
    """
    main()
    print("完成训练和测试")










from torchvision import datasets, transforms
# 下载数据集
# train_set = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
# test_set = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.nn import Conv2d, Linear, ReLU
from torch.nn import MaxPool2d
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# 预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
# 训练数据集
train_data = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# 测试数据集
test_data = MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度：{}".format(train_data_size))
print("测试数据集的长度：{}".format(test_data_size))


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = MaxPool2d(2)
        self.linear1 = Linear(320, 128)
        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 10)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


# 初始化模型、损失函数和优化器
model = MnistModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.14)

# 用于记录训练过程的列表
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


# 模型训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 计算整个epoch的平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    return epoch_loss, epoch_acc


# 模型测试函数
def test(epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 计算整个测试集的平均损失和准确率
    epoch_loss = test_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    test_losses.append(epoch_loss)
    test_accuracies.append(epoch_acc)

    print(f'测试集结果 - 轮次: {epoch}, 平均损失: {epoch_loss:.4f}, '
          f'准确率: {correct}/{total} ({epoch_acc:.2f}%)\n')

    return epoch_loss, epoch_acc


# 加载模型
if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load("./model/model.pkl"))
    print("加载已保存的模型参数")

# 训练与测试
epochs = 10
print("开始训练...")

for epoch in range(1, epochs + 1):
    print(f"————————第{epoch}轮开始——————")

    # 训练
    train_loss, train_acc = train(epoch)
    print(f"训练结果 - 轮次: {epoch}, 平均损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")

    # 测试
    test_loss, test_acc = test(epoch)

    # 保存模型
    torch.save(model.state_dict(), "./model/model.pkl")
    torch.save(optimizer.state_dict(), "./model/optimizer.pkl")

# 7. 可视化训练过程
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train', marker='o')
plt.plot(range(1, len(test_losses) + 1), test_losses, 'r-', label='Test', marker='s')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Train and test loss')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='Train', marker='o')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'r-', label='Test', marker='s')
plt.xlabel('epoch')
plt.ylabel(' Accuracy(%)')
plt.title('Train and test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('./training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 最终测试结果
print("最终测试结果:")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

final_accuracy = 100. * correct / total
print(f"最终测试准确率: {correct}/{total} ({final_accuracy:.2f}%)")

if final_accuracy > 90:
    print("✓ 达到目标: 测试准确率 > 90%")
else:
    print("✗ 未达到目标: 测试准确率 < 90%")

# print(f"\n训练损失记录: {train_losses}")
# print(f"训练准确率记录: {train_accuracies}")
# print(f"测试损失记录: {test_losses}")
# print(f"测试准确率记录: {test_accuracies}")

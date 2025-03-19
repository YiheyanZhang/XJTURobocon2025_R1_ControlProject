import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# ========== 参数配置 ==========
batch_size = 64               # 训练批大小
num_epochs = 2000              # 训练轮数

# ========== 模型定义 ==========
class BallisticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.layers(x)
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # ========== 数据预处理 ==========
    # [:,0]v    [:,1]d
    raw_data = np.loadtxt("data/train_data.txt", delimiter=",")

    # 数据集分割    print(f"理论速度：{theoretical_speed:.2f} m/s")
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data[:,1], raw_data[:,0], test_size=0.2, random_state=42
    )

    # print(format(X_train), format(y_train))
        # 标准化处理
    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    # 转换为PyTorch张量
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train.reshape(-1,1)), 
        torch.FloatTensor(y_train.reshape(-1,1))
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test.reshape(-1,1)), 
        torch.FloatTensor(y_test.reshape(-1,1))
    )
    # ========== 训练配置 ==========
    model = BallisticNet()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = nn.HuberLoss()  # 鲁棒损失函数

    writer = SummaryWriter('log_train')  # 训练日志

    model = model.to(device)
    criterion = criterion.to(device)
    # ========== 训练循环 ==========
    best_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for inputs, targets in DataLoader(train_dataset, batch_size, shuffle=True):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch)
        
        # 验证阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in DataLoader(test_dataset, batch_size):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
            writer.add_scalar('Loss/test', test_loss/len(test_dataset), epoch)
        
        # 学习率调整
        avg_test_loss = test_loss / len(test_dataset)
        scheduler.step(avg_test_loss)
        
        # 保存最佳模型
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), "model/best_model.pth")
        
        # 打印进度
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss/len(train_dataset):.4f} | "
                f"Test Loss: {avg_test_loss:.4f}")
    np.save("model/X_mean.npy", X_mean)
    np.save("model/X_std.npy", X_std)
    np.save("model/y_mean.npy", y_mean)
    np.save("model/y_std.npy", y_std)
    writer.close()

if __name__ == "__main__":
    main()


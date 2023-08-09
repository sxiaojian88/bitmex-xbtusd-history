import torch
import torch.nn as nn
from Network import Network
from Dataset import MyDataset
from torch.utils.data import DataLoader
from datetime import datetime, timedelta

class ModelTrainer:

    def __init__(self, model, train_set, val_set, batch_size, learning_rate):
        self.model = model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)

        self.train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=batch_size)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self._train_one_epoch()
            val_loss = self._validate()
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

    def _train_one_epoch(self):
        self.model.train()
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)

            pred1, pred2 = self.model(X)
            y_combined = torch.cat((y1, y2), dim=1)
            loss = self.criterion(torch.cat((pred1, pred2), dim=1), y_combined)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _validate(self):
        self.model.eval()
        val_loss = 0.0
        for X, y1, y2 in self.val_loader:
            X, y1, y2 = X.to(self.device), y1.to(
                self.device), y2.to(self.device)

            with torch.no_grad():
                pred1, pred2 = self.model(X)
                y_combined = torch.cat((y1, y2), dim=1)
                loss = self.criterion(
                    torch.cat((pred1, pred2), dim=1), y_combined)
                val_loss += loss.item()

        val_loss /= len(self.val_loader)
        return val_loss


# 设置训练参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100

# 定义起止时间
START_TIME = datetime(2021, 1, 1)
END_TIME = START_TIME + timedelta(days=10 * 30) 

# 创建训练集
train_set = MyDataset(START_TIME, END_TIME)

# 创建验证集 
VAL_TIME = END_TIME + timedelta(days=2 * 30)
val_set = MyDataset(END_TIME, VAL_TIME)

# 后续代码...

# 初始化trainer
model = Network()
trainer = ModelTrainer(
    model=model,
    train_set=train_set,
    val_set=val_set,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE
)

# 传递参数进行训练
trainer.train(num_epochs=NUM_EPOCHS)

# load data
import numpy as np
import pickle

# pytorch
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

batch_size = 128
num_classes = 10
epochs = 5

# The data, shuffled and split between train and test sets:
(x_train, y_train) = pickle.load(open('data/big_cifar_data.pkl', 'rb'))
y_train = y_train.flatten()
print('x_train shape:', x_train.shape)
print(y_train[:5])

y_train = y_train.astype('int64')  # for long tensor
x_train = x_train.astype('float32')
x_train = x_train / 127.5
x_train = x_train - 1
# change shape for pytorch
x_train = np.transpose(x_train, [0, 3, 2, 1])
print('new x_train shape:', x_train.shape)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

test_cnt = int(len(x_train) * 0.2)
x_test = x_train[:test_cnt]
y_test = y_train[:test_cnt]


class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.dropout(out, p=0.25)

        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2)
        out = F.dropout(out, p=0.25)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out


train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False, num_workers=4)
net = CNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

x_test = Variable(x_test, volatile=True)
y_test = Variable(y_test)

# train
for i in range(epochs):
    print("epoch", i + 1)
    net.train()
    net.float()
    train_loss = 0
    correct = 0
    total = 0
    start_t = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # cal test loss
    net.eval()
    test_out = net(x_test)
    test_loss = criterion(test_out, y_test).data[0]

    end_t = time.time()
    time_cost = end_t - start_t
    time_cost = round(time_cost, 4)

    train_acc = round(1.0 * correct / total, 4)
    train_loss = train_loss / (batch_idx + 1)

    print("time cost", time_cost, "train loss", round(train_loss, 4), "train acc", train_acc)
    print("test loss", round(test_loss, 4))
    print("------------------------")

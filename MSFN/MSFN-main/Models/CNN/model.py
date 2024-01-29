import torch
import torch.nn as nn
import torch.nn.init as init


class CNN_Clin(nn.Module):
    def __init__(self):
        super(CNN_Clin, self).__init__()

        self.feature_matrices = []
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=25, kernel_size=15, stride=2, padding=(15 - 1) // 2)
        self.Tan = nn.Tanh()
        self.dense1 = nn.Linear(425, 150)
        self.dropout=0.5
        self.output = nn.Linear(150, 1)
        self.Sig = nn.Sigmoid()

    def init_weights(self):
        init.xavier_normal_(self.conv1.weight)
        init.constant_(self.conv1.bias, 0.1)

        init.xavier_normal_(self.dense1.weight)
        init.constant_(self.dense1.bias, 0.1)

        init.xavier_normal_(self.output.weight)
        init.constant_(self.output.bias, 0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Tan(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        feature=x
        x = self.Tan(x)
        x = self.output(x)
        x = self.Sig(x)

        return x,feature

class CNN_CNV(nn.Module):
    def __init__(self):
        super(CNN_CNV, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=15, stride=2, padding=(15 - 1) // 2)
        self.Tan = nn.Tanh()
        self.dense1 = nn.Linear(400, 150)
        self.dropout1=nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.25)
        self.output = nn.Linear(150, 1)
        self.Sig = nn.Sigmoid()

    def init_weights(self):
        init.xavier_normal_(self.conv1.weight)
        init.constant_(self.conv1.bias, 0.1)

        init.xavier_normal_(self.dense1.weight)
        init.constant_(self.dense1.bias, 0.1)

        init.xavier_normal_(self.output.weight)
        init.constant_(self.output.bias, 0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Tan(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.dense1(x)
        feature = x
        x = self.Tan(x)
        x = self.dropout2(x)
        x = self.output(x)
        x = self.Sig(x)

        return x,feature

class CNN_mRNA(nn.Module):
    def __init__(self):
        super(CNN_mRNA, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=15, stride=2, padding=(15 - 1) // 2)
        self.Tan = nn.Tanh()
        self.dense1 = nn.Linear(800, 150)
        self.dropout1=nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.25)
        self.output = nn.Linear(150, 1)
        self.Sig = nn.Sigmoid()

    def init_weights(self):
        init.xavier_normal_(self.conv1.weight)
        init.constant_(self.conv1.bias, 0.1)

        init.xavier_normal_(self.dense1.weight)
        init.constant_(self.dense1.bias, 0.1)

        init.xavier_normal_(self.output.weight)
        init.constant_(self.output.bias, 0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Tan(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.dense1(x)
        feature = x
        x = self.Tan(x)
        x = self.dropout2(x)
        x = self.output(x)
        x = self.Sig(x)

        return x,feature





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.resNet import ResNet18, ResNet34

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.fc = nn.Linear(4096, 1)

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)

        score = self.fc(torch.abs(output1 - output2))

        return score

    def get_embedding(self, x):
        return self.embedding_net(x)
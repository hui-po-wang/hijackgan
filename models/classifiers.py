# +
import torch
import torch.nn as nn
import torchvision.models as models

import os
import numpy as np


# -

class UnifiedRegressor(nn.Module):
    def __init__(self, out_dim):
        super(UnifiedRegressor, self).__init__()
        self.model = nn.Sequential(nn.Linear(512, 256),
                 nn.ReLU(),
                 nn.Linear(256, 100),
                 nn.ReLU(),
                 nn.Linear(100, 100),
                 nn.ReLU(),
                 nn.Linear(100, 100),
                 nn.ReLU(),
                 nn.Linear(100, 100),
                 nn.ReLU(),
                 nn.Linear(100, out_dim))
    
    def forward_index(self, x, index):
        return self.models[index](x) 
    
    def forward(self, x):
        return self.model(x)


class UnifiedDropoutRegressor(nn.Module):
    def __init__(self, out_dim=3):
        super(UnifiedDropoutRegressor, self).__init__()
        self.out_dim = out_dim
        self.model = nn.Sequential(nn.Linear(512, 512//2),
                 nn.ReLU(),
                 nn.Dropout(0.2),
                 nn.Linear(512//2, 100),
                 nn.ReLU(),
                 nn.Dropout(0.2),
                 nn.Linear(100, 100),
                 nn.ReLU(),
                 nn.Dropout(0.2),
                 nn.Linear(100, 100),
                 nn.ReLU(),
                 nn.Dropout(0.2),
                 nn.Linear(100, 100),
                 nn.ReLU(),
                 nn.Dropout(0.2),
                 nn.Linear(100, 100),
                 nn.ReLU(),
                 nn.Dropout(0.2),
                 nn.Linear(100, 100),
                 nn.ReLU(),
                 nn.Dropout(0.2),
                 nn.Linear(100, 100),
                 nn.ReLU(),
                 nn.Dropout(0.2),
                 nn.Linear(100, out_dim)) #yaw, pitch, roll
    
    def forward_index(self, x, index):
        return self.models[index](x) 
    
    def forward(self, x):
        return self.model(x)


class DropoutCompoundClassifier(nn.Module):
    def __init__(self, attr_num, if_bn=False):
        super(DropoutCompoundClassifier, self).__init__()
        self.attr_num = attr_num
        self.models = nn.ModuleList()
        for i in range(attr_num):
            
            self.models.append(nn.Sequential(nn.Linear(512, 256),
                     nn.ReLU(),
                     nn.Dropout(p=0.5),
                     nn.Linear(256, 100),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),
                     nn.Linear(100, 100),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),
                     nn.Linear(100, 100),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),
                     nn.Linear(100, 100),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),
                     nn.Linear(100, 100),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),
                     nn.Linear(100, 100),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),
                     nn.Linear(100, 100),
                     nn.ReLU(),
                     nn.Dropout(p=0.5),
                     nn.Linear(100, 1)))
    
    def forward_index(self, x, index):
        return self.models[index](x) 
    
    def forward(self, x):
        out = []
        for i in range(self.attr_num):
            out_i = self.models[i](x)
            out.append(out_i)
            
        return torch.cat(out, 1)

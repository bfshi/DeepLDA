
# coding: utf-8

# In[13]:


import torch
import torch.nn as nn



class DeepLDA(nn.Module):
    def __init__(self):
        super(DeepLDA, self).__init__()
        
        self.layer = nn.Sequential(
        nn.Conv2d(3, channel_1, 3, padding = 1),
        nn.BatchNorm2d(channel_1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(channel_1, channel_2, 3, padding = 1), 
        nn.BatchNorm2d(channel_2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(channel_2, channel_3, 3, padding = 1), 
        nn.BatchNorm2d(channel_3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        #16 * 4 * 4 = 256 dims
        #nn.Linear(16 * 4 * 4, 10)
        )
        
        
    def forward(self, x):
        H = self.layer(x)
        return H







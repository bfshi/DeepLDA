
# coding: utf-8

# In[13]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import torch.cuda as cuda

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin


# In[2]:


USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)


# # Load Dataset

# In[5]:


#加载数据集，分为大小为32的batch，对每张图片做mean normalization
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))])
trainset = torchvision.datasets.CIFAR10(root = '/m/shibf/dataset/cifar10', transform = transform)
testset = torchvision.datasets.CIFAR10(root = '/m/shibf/dataset/cifar10', train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size=12000, shuffle = True)


# # Network

# In[6]:


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


# In[7]:


channel_1 = 64
channel_2 = 32
channel_3 = 16

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


# In[65]:


torch.Tensor().shape


# In[87]:


categ = 10
eig_num = 4
lamb = 0.01
epsilon = 1

def LDAloss(H, label):
    N, C, d, _ = H.shape
    N_new = N * d * d
    H = H.permute(0, 2, 3, 1)
    H = torch.reshape(H, (N_new, C))
    H_bar = H - torch.mean(H, 0, True)
    label = label.view(N, 1)
    labels = torch.reshape(label * torch.Tensor().new_ones((N, d * d), device=device, dtype=torch.long), (N_new,))
    S_w = torch.Tensor().new_zeros((C, C), device = device, dtype = dtype)
    S_t = H_bar.t().matmul(H_bar) / (N_new - 1)
    for i in range(categ):
        H_i = H[torch.nonzero(labels == i).view(-1)]
        H_i_bar = H_i - torch.mean(H_i, 0, True)
        N_i = H_i.shape[0]
        if N_i == 0:
            continue
        S_w += H_i_bar.t().matmul(H_i_bar) / (N_i - 1) / categ
    temp = (S_w + lamb * torch.diag(torch.Tensor().new_ones((C), device = device, dtype = dtype))).pinverse().matmul(S_t - S_w)
    w, v = torch.symeig(temp, eigenvectors = True)
    w = w.detach()
    v = v.detach()
    mask = (w >= w[8]) * (w <= (w[8] + epsilon))
    v = v[:, torch.nonzero(mask).view(-1)]
    loss = (v.t().matmul(temp).matmul(v)).sum()
    return -loss / eig_num, w


# # Train

# In[ ]:


def check_LDA(loader, model, ifpert = False, pert = None):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            
            H = model(x)
            loss, eig_value = LDAloss(H, y)
            print('loss = %.4f, eig_mean8 = %.4f' % (loss.item(), eig_value[8:16].mean()))
            print(eig_value[8 : 16])
            return eig_value[8:16].mean()


# In[88]:


def train(model, optimizer, epochs, best_result):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(trainloader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            H = model(x)
            loss, eig_value = LDAloss(H, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f, eig_mean8 = %.4f' % (t, loss.item(), eig_value[8:16].mean()))
                print(eig_value[8 : 16])
        
        result = check_LDA(testloader, model)
        if result > best_result:
            best_result = result
            torch.save(model, "model_DeepLDA_original_loss")
    return best_result


# In[93]:


model = DeepLDA()
#model = torch.load('model_DeepDLA')
best_result = 0


# In[94]:


print_every = 100
learning_rate = 0.1
epochs = 25

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for i in range(5):
    best_result = train(model, optimizer, epochs, best_result)
    learning_rate /= 2

print("The largest mean of top 8 eigvalues is %.4f" % (best_result))
print("model is saved as \"model_DeepLDA_original_loss\"")


# In[ ]:




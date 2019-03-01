
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

from lib.model import DeepLDA


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
#trainloader0 = torch.utils.data.DataLoader(trainset, batch_size=48000, shuffle = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle = True)


# # Test

# In[ ]:


channel_3 = 16

class FCLayer(nn.Module):
    def __init__(self):
        super(FCLayer, self).__init__()
        
        self.layer = nn.Sequential(
        nn.Linear(channel_3, 10)
        )
        
        
    def forward(self, x):
        H = self.layer(x)
        return H


# In[ ]:


base_num = 8

def GetBaseVector(H, label):
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
    return v[:, base_num:]


# In[ ]:


def check_accuracy(loader, base_model, model):
    '''
    return counts of predictions as targetlabel
    '''
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    base_model.eval()
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            H = base_model(x)
            H = torch.mean(H, (2, 3))
            scores = model(H)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc


# In[ ]:


def train(base_model, model, optimizer, epochs):
    best_acc = 0
    base_model = base_model.to(device = device)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(trainloader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            
            with torch.no_grad():
                H = base_model(x)
                H = torch.mean(H, (2, 3))

            scores = model(H)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                print()
        acc = check_accuracy(testloader, base_model, model)
        if (acc > best_acc):
            best_acc = acc
            torch.save(model, "model_FCLayer")
    return best_acc


# In[1]:


base_model = torch.load("model_DeepLDA")
base_model = base_model.to(device = device)
base_model.eval()

'''
for t, (x, y) in trainloader0:
    with torch.no_grad():
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)
        H = base_model(x)
        base = GetBaseVector(H, y)
        break
'''

model = FCLayer()

print_every = 100
learning_rate = 1e-4
epochs = 10
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

best_acc = train(base_model, model, optimizer, epochs)
print("The largest accuracy is %.4f" % (best_acc))
print("model is saved as \"model_FCLayer\"")


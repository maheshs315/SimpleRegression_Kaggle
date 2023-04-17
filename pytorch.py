# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:39:03 2023

@author: S Mahesh Reddy
"""
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib


dataset=pd.read_csv('train.csv')
test=pd.read_csv('train.csv')


dataset = dataset.dropna()


#x_numpy=dataset['x'].values.reshape(-1,1)
#y_numpy=dataset['y'].values.reshape(-1,1)

x_train=dataset['x'].to_numpy().reshape(-1,1)
y_train=dataset['y'].to_numpy().reshape(-1,1)

x_train = dataset.iloc[:,0].values.reshape(-1,1)
y_train = dataset.iloc[:,1].values.reshape(-1,1)


x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

dataset_train = TensorDataset(x_train_tensor, y_train_tensor)


# how assign
train_loader = DataLoader(dataset=dataset_train, batch_size=50)


def make_train_step(model, loss_fn, optimizer):
    # builds & returns the function that will be called inside the loop
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step


device = 'cpu'

# hyperparameters
lr = 1e-6
n_epochs = 1000

from sklearn.metrics import r2_score

# loss function & optimizer
model = nn.Sequential(nn.Linear(1, 1)).to(device)
loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# training step
train_step = make_train_step(model, loss_fn, optimizer)
training_losses = []
test_losses = []
accuracies = []
for epoch in range(n_epochs):
    batch_losses = []
    for nbatch, (x_batch, y_batch) in enumerate(train_loader):
        #print(nbatch)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = train_step(x_batch, y_batch)
        batch_losses.append(loss)
    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)
    # 
    
    if epoch % 50 == 0:
        print(f'epoch {epoch+1} | Training loss: {training_loss:.4f} ')
        
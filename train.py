import numpy as np
from model import CNN
from dataset import train_loader, test_loader
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
import torch

model = CNN()
optimizer = Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
# number of epochs to train the model
n_epochs = 32
train_loss_list = []
valid_loss_list = []
dice_score_list = []
lr_rate_list = []
valid_loss_min = np.Inf  # track change in validation loss
for epoch in tqdm(range(1, n_epochs + 1)):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    dice_score = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    # bar = tq(train_loader, postfix={"train_loss": 0.0})
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # print(loss)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)
        # bar.set_postfix(ordered_dict={"train_loss": loss.item()})

    # calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    train_loss_list.append(train_loss)

    # print training/validation statistics
    print(f'Epoch: {epoch}  Training Loss: {train_loss}')
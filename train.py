import os
import numpy as np
from model import UNetLight
from dataset import train_loader, val_loader
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
import torch
from datetime import datetime
from utils import accuracy

model = UNetLight(n_channels=train_loader.dataset.num_channels, n_classes=train_loader.dataset.num_classes, k=64)
optimizer = Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)
loss_fcn = nn.CrossEntropyLoss(weight=train_loader.dataset.class_weights())
save_path = os.path.join('saved_models', str(datetime.now()) + '.pt')
n_epochs = 15
plot_every = 10
train_loss_list = []
valid_loss_list = []
train_acc_list = []
val_acc_list = []
fig_train, fig_val = None, None
lr_rate_list = []
valid_loss_min = np.Inf  # track change in validation loss

epoch_bar = tqdm(range(1, n_epochs + 1), desc=' Epochs')
for epoch in epoch_bar:
    # train
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        batch_loss = loss_fcn(output, target)
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item() * data.shape[0]
        batch_acc = accuracy(output, target)
        train_acc += batch_acc * data.shape[0]

    # validate
    val_loss = 0.0
    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            batch_loss = loss_fcn(output, target)
            val_loss += batch_loss.item() * data.shape[0]
            batch_acc = accuracy(output, target)
            val_acc += batch_acc * data.shape[0]

    # calculate average losses and append to lists
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)
    val_acc = val_acc / len(val_loader.dataset)
    train_loss_list.append(train_loss)
    valid_loss_list.append(val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    epoch_bar.write('')
    epoch_bar.write('epoch: {}  train_loss: {:.3f}  val_loss: {:.3f} train acc: {:.3f}  val_acc: {:.3f}'
                    .format(epoch, train_loss, val_loss, train_acc, val_acc))

    # save model (if improved on val set)
    if val_loss <= valid_loss_min:
        epoch_bar.write('Validation loss decreased ({:.3f} --> {:.3f}).  Saving model ...'
                        .format(valid_loss_min, val_loss))
        torch.save(model.state_dict(), save_path)
        valid_loss_min = val_loss

    scheduler.step(val_loss)

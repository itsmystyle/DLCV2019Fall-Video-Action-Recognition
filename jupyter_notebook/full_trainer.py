#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import pickle
import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import skimage
# from tqdm.notebook import tqdm
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from modules.dataset import FullLengthVideosDataset
from modules.models.LRCNN import SeqRecurrentCNN
from modules.metrics import MulticlassAccuracy
from modules.utils import set_random_seed

rescale_factor = 1.0
random_seed = 42

set_random_seed(random_seed)
save_dir = "../models/Full_LRCNN/"
full_path = "/tmp2/itsmystyle/h4_data/FullLengthVideos/"
train_video_path = os.path.join(full_path, 'videos', 'train')
train_label_path = os.path.join(full_path, 'labels', 'train')
valid_video_path = os.path.join(full_path, 'videos', 'valid')
valid_label_path = os.path.join(full_path, 'labels', 'valid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


# hyper-parameters
epochs = 100
lr = 2e-4
accumulate_gradient = 8

model = SeqRecurrentCNN()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)

criterion = nn.NLLLoss()

metric = MulticlassAccuracy()

writer = SummaryWriter(os.path.join(save_dir, "train_logs"))


# In[ ]:


train_set = FullLengthVideosDataset(train_video_path, train_label_path, length=80, overlap=20, sorting=True)
valid_set = FullLengthVideosDataset(valid_video_path, valid_label_path, length=100, overlap=0, sorting=True)


# In[ ]:


train_loader = DataLoader(train_set, batch_size=4, num_workers=32, shuffle=True, collate_fn=train_set.collate_fn)
valid_loader = DataLoader(valid_set, batch_size=4, num_workers=32, shuffle=False, collate_fn=valid_set.collate_fn)


# In[ ]:


def _run_one_epoch(epoch, iters):
    model.train()
    
    trange = tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch {}".format(epoch))
    
    metric.reset()
    batch_loss = 0.0
    
    for idx, batch in trange:
        iters += 1

        frames = batch["frames"].to(device)
        frames_len = batch["frames_len"]
        labels = batch["labels"].to(device)

        preds = model(frames, frames_len)

        # calculate loss and update weights
        loss = criterion(preds.view(-1, preds.shape[-1]), labels.view(-1)) / accumulate_gradient
        if idx % accumulate_gradient == 0:
            optimizer.zero_grad()
        loss.backward()
        if (idx + 1) % accumulate_gradient == 0:
            optimizer.step()

        # update metric
        metric.update(preds.view(-1, preds.shape[-1]).cpu(), labels.view(-1).cpu())

        # update loss
        batch_loss += loss.item() * accumulate_gradient
        writer.add_scalars(
            "Loss", {"iter_loss": loss.item(), "avg_loss": batch_loss / (idx + 1)}, iters,
        )

        # update tqdm
        trange.set_postfix(
            loss=batch_loss / (idx + 1), **{metric.name: metric.print_score()}
        )

    if (idx + 1) % accumulate_gradient != 0:
        optimizer.step()
        optimizer.zero_grad()

    return batch_loss / (idx + 1), iters


# In[ ]:


def _eval_one_epoch(val_iters, best_accuracy):
    model.eval()

    trange = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Valid")

    metric.reset()
    batch_loss = 0.0

    with torch.no_grad():
        for idx, batch in trange:
            val_iters += 1

            frames = batch["frames"].to(device)
            frames_len = batch["frames_len"]
            labels = batch["labels"].to(device)

            preds = model(frames, frames_len)
            loss = criterion(preds.view(-1, preds.shape[-1]), labels.view(-1))

            # update loss
            batch_loss += loss.item()
            writer.add_scalars(
                "Val_Loss",
                {"iter_loss": loss.item(), "avg_loss": batch_loss / (idx + 1)},
                val_iters,
            )

            # update metric
            metric.update(preds.view(-1, preds.shape[-1]).cpu(), labels.view(-1).cpu())

            # update tqdm
            trange.set_postfix(
                loss=batch_loss / (idx + 1), **{metric.name: metric.print_score()}
            )

        # save best acc model
        if metric.get_score() > best_accuracy:
            print("Best model saved!")
            best_accuracy = metric.get_score()
            _loss = batch_loss / (idx + 1)
            torch.save(
                model.state_dict(), os.path.join(
                    save_dir,
                    "model_best_{:.5f}_{:.5f}.pth.tar".format(best_accuracy, _loss),
                )
            )

    return batch_loss / (idx + 1), best_accuracy, val_iters


# In[ ]:


iters = -1
val_iters = -1
best_accuracy = 0.0

for epoch in range(epochs + 1):
    
    loss, iters = _run_one_epoch(epoch, iters)
    
    loss, best_accuracy, val_iters = _eval_one_epoch(val_iters, best_accuracy)


# In[ ]:





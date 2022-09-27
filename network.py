from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy


class Network(pl.LightningModule):
    def __init__(self, c1: int, ou: int, e: float = 0.5, n: int = 5):
        super(Network, self).__init__()
        c_ = int(c1 * e) * 4
        self.save_hyperparameters()
        self.acc_val = Accuracy()
        self.acc_train = Accuracy()
        self.e = nn.Sequential(*(nn.Sequential(nn.Linear(c1 if i == 0 else c_, c_), nn.ReLU()) for i in range(n)))
        self.u = nn.Linear(c_, ou)
        self.s = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        print(self.parameters())

    def forward(self, x):
        return self.s(self.u(self.e(x.float().clone().detach().requires_grad_(True))))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 0.0001)
        lr_lambda = lambda epoch: 0.85 ** epoch
        lr_scheduler = optim.lr_scheduler.LambdaLR(lr_lambda=lr_lambda, optimizer=optimizer)
        # self.log('Lr', lr_scheduler.get_lr()[0], on_epoch=True, prog_bar=True)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_index):
        x, y = batch
        bs = x.shape[0]
        ld = y.shape[-1]
        x_ = self(x)
        x_ = x_.view(bs, ld)
        y = y.view(bs, ld)
        loss = self.loss(x_, y)
        acc = self.acc_train(x_, y.int())
        self.log('train_loss', loss.item())
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_index):
        x, y = batch
        bs = x.shape[0]
        ld = y.shape[-1]
        x_ = self(x)
        x_ = x_.view(bs, ld)
        y = y.view(bs, ld)
        loss = self.loss(x_, y)
        acc = self.acc_val(x_, y.int())
        self.log('val_loss', loss.item())
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return x_

    # def backward(
    #         self, loss, optimizer, optimizer_idx, *args, **kwargs
    # ) -> None:
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

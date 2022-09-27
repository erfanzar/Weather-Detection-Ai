import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from network import Network
from utils.dataloader import CustomDataSet
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, RichProgressBar, ModelCheckpoint, \
    BackboneFinetuning

global name_, out_

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    mcp = ModelCheckpoint(dirpath='/save/', save_on_train_epoch_end=True, save_top_k=2, monitor='val_loss')

    data_set_train = CustomDataSet('wd.csv', val=False)
    data_set_val = CustomDataSet('wd.csv', val=True)

    train_loader = DataLoader(dataset=data_set_train, batch_size=64, shuffle=True, num_workers=2)
    validation_loader = DataLoader(dataset=data_set_val, batch_size=64, shuffle=False, num_workers=2)
    lr = lambda epoch: 1.5
    # backbone = BackboneFinetuning(10, lambda_func=lr, )
    ckpt_save = ModelCheckpoint(dirpath='model/save/', save_on_train_epoch_end=True, save_top_k=-1, monitor='train_acc',
                                filename='ckpt',
                                )
    early_stop = EarlyStopping(monitor='val_loss', mode='min')
    lr_monitor = LearningRateMonitor()
    trainer = pl.Trainer(accelerator='gpu' if DEVICE == 'cuda:0' else 'cpu', min_epochs=None, max_epochs=5000,
                         auto_lr_find=True, enable_checkpointing=True,
                         callbacks=[mcp, early_stop, lr_monitor,
                                    # backbone,
                                    ckpt_save])

    net = Network(6, 49)
    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    trainer.save_checkpoint('best_ckpt.ckpt')

    # x, y = next(iter(data_loader))
    # print(y)

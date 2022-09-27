import torch
import torch.nn as nn

import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from network import Network

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    trainer = pl.Trainer(accelerator='gpu' if DEVICE == 'cuda:0' else 'cpu', min_epochs=None, max_epochs=5000,
                         auto_lr_find=True, enable_checkpointing=True,
                         )

    net = Network(6, 49)
    script = net.to_torchscript()
    input_sample = torch.randn((1, 6))
    onnx = net.to_onnx('model.onnx', input_sample, export_params=True)
    torch.jit.save(script, 'model.pt')

    # x, y = next(iter(data_loader))
    # print(y)

import sys
from abc import ABC

import pandas
import torch

from torch.utils.data import DataLoader, Dataset

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class CustomDataSet(Dataset, ABC):
    def __init__(self, path: str, val: bool = False):
        super(CustomDataSet, self).__init__()
        self.data = pandas.read_csv(path)
        self.x = []
        self.y = []
        self.vd = {}
        self.val = val
        self.start()
        self.ct = len(self.y) - int(len(self.y) / 4) if self.val is False else int(len(self.y) / 4)

    def start(self):

        date_time, temp, dew, rel, wind, visibility, press, weather = self.data
        date_time, temp, dew, rel, wind, visibility, press, weather = [
            self.data[date_time].values,
            self.data[temp].values,
            self.data[dew].values,
            self.data[rel].values,
            self.data[wind].values,
            self.data[visibility].values,
            self.data[press].values,
            self.data[weather].values
        ]
        tti = 1
        nma = "Validation" if self.val else "Training "
        for index, o in enumerate(weather):
            tti += 1
            t, d, r, w, v, p = temp[index] / max(temp), dew[index] / max(dew), rel[index] / max(rel), wind[index] / max(
                wind), visibility[index] / max(visibility), press[index] / max(press)
            x = torch.tensor([t, d, r, w, v, p])
            self.x.append(x)
            if o not in self.vd:
                self.vd[o] = len(self.vd)
            y_o = [val for k, val in self.vd.items() if k == o]
            self.y.append(torch.tensor(y_o[0]))
            sys.stdout.write(
                f'\r {nma} -> Writing Data Onto {"GPU" if DEVICE == "cuda:0" else "Ram"} '
                f' % {int((tti / len(date_time)) * 100)}')
        # print('\n')

    def __len__(self):

        return self.ct

    def __getitem__(self, item):
        item += len(self.y) - int(len(self.y) / 4) if self.val else 0
        y = torch.zeros((1, 49))
        y[0, self.y[item] - 1] = 1

        return self.x[item], y

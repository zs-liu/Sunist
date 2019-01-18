import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SunistDataset(Dataset):

    def __init__(self, train, data_root, label_root, number):
        self.train = train
        self.number = number
        self.data_list = []
        self.label_list = []
        print("Reading data and label.")
        for i in range(0, self.number):
            data = cv2.imread(data_root + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(label_root + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
            data = np.array(data).reshape(1, 500, 500).astype(np.float32)
            data = data / data.max()
            label = np.array(label == 255).reshape(500, 500).astype(np.int64)
            # label = self.relabel(label)
            self.data_list.append(data)
            self.label_list.append(label)
        print("Reading finish.")

    def __getitem__(self, item):
        data = self.data_list[item]
        label = self.label_list[item]
        return data, label

    def __len__(self):
        return len(self.data_list)

    def relabel(self, label):
        new_label = np.zeros_like(label)
        for i in range(1, 499):
            for j in range(1, 499):
                if label[i][j] == 1 and (label[i - 1, j] == 0 or label[i, j - 1] == 0 or
                                         label[i + 1, j] == 0 or label[i, j + 1] == 0):
                    new_label[i][j] = 1
        return new_label

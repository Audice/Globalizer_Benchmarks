from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array, shape (N, 12, L) or (N, 1, L)
        y: numpy array, shape (N,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.preds = []
        self.labels = []
        self.probs = []

    def update(self, loss, preds, labels, probs=None):
        self.losses.append(loss)
        self.preds.extend(preds.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        if probs is not None:
            self.probs.extend(probs.detach().cpu().numpy())

    def compute(self):
        y_true = np.array(self.labels)
        y_pred = np.array(self.preds)

        metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
        }

        if len(self.probs) > 0:
            try:
                metrics['auc'] = roc_auc_score(y_true, np.array(self.probs)[:, 1])
            except:
                metrics['auc'] = 0.0

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.update({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
        except:
            metrics.update({'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0})

        return metrics


class WeightedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, alpha=0.25, gamma=2.0, beta=0.999):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.register_buffer('cb_weights', torch.FloatTensor(weights))

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        cb_weights = self.cb_weights[targets]
        return (focal_loss * cb_weights).mean()
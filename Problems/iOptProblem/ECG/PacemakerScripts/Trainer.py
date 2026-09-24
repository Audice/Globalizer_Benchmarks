import os
from tqdm import tqdm
from ECG.PacemakerScripts.MetricsCalculator import *
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, model, device, train_id, save_dir='checkpoints'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.train_id = train_id
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        metrics = MetricsCalculator()

        for signals, labels in tqdm(loader, desc='Training'):
            signals, labels = signals.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            metrics.update(loss.item(), preds, labels, F.softmax(outputs, dim=1))

        return metrics.compute()

    def validate(self, loader, criterion):
        self.model.eval()
        metrics = MetricsCalculator()

        with torch.no_grad():
            for signals, labels in tqdm(loader, desc='Validation'):
                signals, labels = signals.to(self.device), labels.to(self.device)
                outputs = self.model(signals)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                metrics.update(loss.item(), preds, labels, F.softmax(outputs, dim=1))

        return metrics.compute()

    def train(self, train_loader, val_loader, criterion, optimizer,
              epochs=100, patience=20, scheduler=None):
        best_target_metric = 0
        best_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            val_metrics = self.validate(val_loader, criterion)

            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1'])

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()

            if val_metrics['mcc'] > best_target_metric:
                best_target_metric = val_metrics['mcc']
                best_state = self.model.state_dict().copy()
                patience_counter = 0
                torch.save(best_state, os.path.join(self.save_dir, f"{self.train_id}.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Ранняя остановка: {epoch + 1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        return history, best_target_metric

    def test(self, test_loader, criterion):
        return self.validate(test_loader, criterion)
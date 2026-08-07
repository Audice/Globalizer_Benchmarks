import numpy as np


class ImbalancedMetrics:
    @staticmethod
    def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
        """
        Вычисляет метрики для бинарной классификации
        """
        y_pred_binary = (y_pred > threshold).astype(np.float32)
        y_true = y_true.astype(np.float32)

        # TP, TN, FP, FN
        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true == 0))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))

        eps = 1e-8

        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)

        f1 = 2 * precision * recall / (precision + recall + eps)

        # F-beta (beta=2 - больше внимания recall)
        beta = 2
        f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + eps)

        # Matthews Correlation Coefficient
        mcc_num = tp * tn - fp * fn
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
        mcc = mcc_num / mcc_den

        # G-mean
        g_mean = np.sqrt(recall * specificity)

        # Balanced Accuracy
        balanced_acc = (recall + specificity) / 2

        # Классовые метрики
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1': float(f1),
            'f_beta': float(f_beta),
            'mcc': float(mcc),
            'g_mean': float(g_mean),
            'balanced_accuracy': float(balanced_acc),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }

        # Добавляем метрики для каждого класса
        # Для класса 0 (norm)
        metrics['class_0_precision'] = tn / (tn + fn + eps)
        metrics['class_0_recall'] = tn / (tn + fp + eps)
        metrics['class_0_f1'] = 2 * metrics['class_0_precision'] * metrics['class_0_recall'] / (
                    metrics['class_0_precision'] + metrics['class_0_recall'] + eps)

        # Для класса 1 (epi)
        metrics['class_1_precision'] = metrics['precision']
        metrics['class_1_recall'] = metrics['recall']
        metrics['class_1_f1'] = metrics['f1']

        return metrics

    @staticmethod
    def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
        from sklearn.metrics import classification_report, confusion_matrix

        y_pred_binary = (y_pred > threshold).astype(np.float32)

        print("Метрики классификации")
        print(classification_report(y_true, y_pred_binary, target_names=['norm', 'epi']))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        print("Confusion Matrix:")
        print(f"              Predicted")
        print(f"              norm    epi")
        print(f"Actual norm   {cm[0, 0]:5d}   {cm[0, 1]:5d}")
        print(f"       epi    {cm[1, 0]:5d}   {cm[1, 1]:5d}")

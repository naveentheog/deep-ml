import numpy as np

def compute_cross_entropy_loss(y_pred, y_true, epsilon=1e-15):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), epsilon, 1 - epsilon)
    y_true = np.asarray(y_true, dtype=float)
    per_sample = -np.sum(y_true * np.log(y_pred), axis=1)  # loss for each row
    return float(np.mean(per_sample))
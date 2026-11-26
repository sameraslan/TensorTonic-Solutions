import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    N = y_pred.shape[0]
    p_true = y_pred[np.arange(N), y_true]
    
    logged = -np.log(p_true)
    loss = float(np.mean(logged))

    return loss
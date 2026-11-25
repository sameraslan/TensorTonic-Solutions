import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    max = np.max(x, axis=-1, keepdims=True)  # Need to do keepdims = true to end up with a 2x1 as opposed to a 1x2 which numpy would autocast to
    exps = np.exp(x - max)
    denom = np.sum(exps, axis=-1, keepdims=True)
    divided_exps = exps / denom
    
    return divided_exps
    

    
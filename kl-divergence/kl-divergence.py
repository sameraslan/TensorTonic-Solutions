import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    clipped_p = np.clip(p, eps, None)
    clipped_q = np.clip(q, eps, None)
    div = clipped_p / clipped_q
    logged = np.log(div)
    mult = p * logged  # Keeping unclipped_p as multiplier, clipped is only so that log doesn't give inf

    summed = np.sum(mult)

    return float(summed)
    
import numpy as np
from numpy.typing import NDArray

def portfolio_var(
        weigths: NDArray[np.float], 
        sigma: NDArray[np.float]
        ) -> np.float:
    '''Function for calculating portfolio variance
    Args:
        weights (NDArray): numpy array with portfolio weights
        sigma (NDArray): covariance matrix
    Returns:
        portfolio_variance (float): portfolio variance
    '''
    portfolio_variance = weigths.T @ sigma @ weigths

    return portfolio_variance
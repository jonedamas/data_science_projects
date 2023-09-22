import numpy as np

from numpy.typing import NDArray
from scipy.optimize import minimize

def portfolio_var(
        weigths: NDArray[np.float64], 
        sigma: NDArray[np.float64]
    ) -> np.float64:
    '''Function for calculating portfolio variance
    Args:
        weights (NDArray): numpy array with portfolio weights
        sigma (NDArray): covariance matrix
    Returns:
        portfolio_variance (float): portfolio variance
    '''
    portfolio_variance = weigths.T @ sigma @ weigths

    return portfolio_variance

def portfolio_sharpe(
        weigths: NDArray[np.float64], 
        sigma: NDArray[np.float64],
        mu: NDArray[np.float64],
        r: float
    ) -> np.float64:

    '''Function for calculating portfolio sharpe ratio
    Args:
        weights (NDArray): numpy array with portfolio weights
        sigma (NDArray): covariance matrix
        mu: mean returns
    Returns:
        portfolio_sharpe (float): portfolio sharpe ratio
    '''
    portfolio_sharpe = (sum(weigths * mu) - r) / np.sqrt(portfolio_var(weigths, sigma))

    return portfolio_sharpe


def find_OW(
        function, 
        *args: NDArray[np.float64], 
        n: int = 1
    ) -> NDArray[np.float64]:
    
    '''Function for calculating portfolio optimal weights
    Args:
        function (func): objective function to be minimized
        *args (NDArray): objective function arguments
        n (int): number of weights

    Returns:
        OW (NDArray): numpy array with optimal portfolio weights
    '''
    res = minimize(
        function, 
        x0=np.ones(n) / n, 
        method='SLSQP', 
        bounds=tuple((0,1) for x in range(n)), 
        constraints = {'type': 'eq', 'fun': lambda x:  1 - sum(x)},
        args=args,
    )
    OW = res.x

    return OW

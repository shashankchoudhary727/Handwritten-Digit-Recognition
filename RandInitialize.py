import numpy as np

def initialise(a, b):
    """
    Initialize weights using He initialization
    
    Parameters:
    a: number of units in current layer
    b: number of units in previous layer
    
    Returns:
    Initialized weight matrix of shape (a, b+1)
    """
    # He initialization for better training with ReLU
    epsilon = np.sqrt(2.0 / b)
    
    # Initialize weights with random values from normal distribution
    weights = np.random.randn(a, b + 1) * epsilon
    
    # Initialize bias terms to small positive values to help with initial learning
    weights[:, 0] = 0.01
    
    return weights
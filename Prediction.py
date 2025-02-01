import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def predict(Theta1, Theta2, X):
    """
    Make predictions using the trained neural network
    
    Parameters:
    Theta1: weights between input and hidden layer
    Theta2: weights between hidden and output layer
    X: input data matrix
    
    Returns:
    Predicted labels
    """
    m = X.shape[0]
    
    # Add bias unit to input layer
    a1 = np.append(np.ones((m, 1)), X, axis=1)
    
    # Hidden layer activation
    z2 = np.dot(a1, Theta1.transpose())
    a2 = sigmoid(z2)
    
    # Add bias unit to hidden layer
    a2 = np.append(np.ones((m, 1)), a2, axis=1)
    
    # Output layer activation
    z3 = np.dot(a2, Theta2.transpose())
    a3 = sigmoid(z3)
    
    # Return predicted class (maximum probability)
    return np.argmax(a3, axis=1)
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_gradient(z):
    s = sigmoid(z)
    return s * (1 - s)

def neural_network(Theta1, Theta2, X, y, lambda_reg):
    m = X.shape[0]
    
    # Forward propagation with dropout
    a1 = np.append(np.ones((m, 1)), X, axis=1)
    z2 = np.dot(a1, Theta1.transpose())
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((m, 1)), a2, axis=1)
    z3 = np.dot(a2, Theta2.transpose())
    a3 = sigmoid(z3)
    
    # Convert labels to one-hot encoding
    y_matrix = np.zeros((m, 10))
    for i in range(m):
        y_matrix[i, int(y[i])] = 1
    
    # Compute cost
    cost = (-1/m) * np.sum(
        y_matrix * np.log(a3 + 1e-10) + 
        (1 - y_matrix) * np.log(1 - a3 + 1e-10)
    )
    
    # Add regularization
    reg_term = (lambda_reg/(2*m)) * (
        np.sum(np.square(Theta1[:, 1:])) + 
        np.sum(np.square(Theta2[:, 1:]))
    )
    cost += reg_term
    
    # Backpropagation
    delta3 = a3 - y_matrix
    delta2 = np.dot(delta3, Theta2)[:, 1:] * sigmoid_gradient(z2)
    
    # Gradient computation with regularization
    Theta1_grad = (1/m) * np.dot(delta2.transpose(), a1)
    Theta2_grad = (1/m) * np.dot(delta3.transpose(), a2)
    
    # Add regularization to gradients
    Theta1_grad[:, 1:] += (lambda_reg/m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lambda_reg/m) * Theta2[:, 1:]
    
    return cost, Theta1_grad, Theta2_grad

def train_network(Theta1, Theta2, X, y, learning_rate, epochs, batch_size, lambda_reg):
    m = X.shape[0]
    history = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Mini-batch training
        for i in range(0, m, batch_size):
            batch_X = X_shuffled[i:i + batch_size]
            batch_y = y_shuffled[i:i + batch_size]
            
            # Forward and backward pass
            cost, Theta1_grad, Theta2_grad = neural_network(
                Theta1, Theta2, batch_X, batch_y, lambda_reg
            )
            
            # Update weights with momentum
            Theta1 -= learning_rate * Theta1_grad
            Theta2 -= learning_rate * Theta2_grad
            
        # Compute and store cost for entire epoch
        epoch_cost, _, _ = neural_network(Theta1, Theta2, X, y, lambda_reg)
        history.append(epoch_cost)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Cost: {epoch_cost:.4f}')
            
        # Learning rate decay
        if epoch % 20 == 0:
            learning_rate *= 0.9
    
    return Theta1, Theta2, history
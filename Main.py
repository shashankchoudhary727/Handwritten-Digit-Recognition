from scipy.io import loadmat
import numpy as np
from Model import neural_network, train_network
from RandInitialize import initialise
from Prediction import predict
import matplotlib.pyplot as plt

# Loading and preprocessing data
print("Loading MNIST dataset...")
data = loadmat('mnist-original.mat')
X = data['data'].transpose()
X = X / 255.0  # Normalize to [0,1]
y = data['label'].flatten()

# Split data
print("Splitting data into train/test sets...")
X_train = X[:60000, :]
y_train = y[:60000]
X_test = X[60000:, :]
y_test = y[60000:]

# Network architecture
input_layer_size = 784  # 28x28 pixels
hidden_layer_size = 200  # Increased from 100
num_labels = 10

# Initialize parameters
print("Initializing network parameters...")
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

# Training parameters
learning_rate = 0.1
epochs = 100
batch_size = 128
lambda_reg = 0.01  # Reduced from 0.1

# Train the network
print("Training neural network...")
Theta1, Theta2, history = train_network(
    initial_Theta1, 
    initial_Theta2, 
    X_train, 
    y_train,
    learning_rate,
    epochs,
    batch_size,
    lambda_reg
)

# Plot training progress
plt.plot(history)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluate model
print("\nEvaluating model performance...")
pred_test = predict(Theta1, Theta2, X_test)
test_accuracy = np.mean(pred_test == y_test) * 100
print('Test Set Accuracy: {:.2f}%'.format(test_accuracy))

pred_train = predict(Theta1, Theta2, X_train)
train_accuracy = np.mean(pred_train == y_train) * 100
print('Training Set Accuracy: {:.2f}%'.format(train_accuracy))

# Calculate precision for each digit
print("\nPer-digit precision:")
for digit in range(10):
    mask = (y_train == digit)
    true_positive = np.sum((pred_train == digit) & mask)
    false_positive = np.sum((pred_train == digit) & ~mask)
    precision = true_positive / (true_positive + false_positive + 1e-10)
    print(f'Digit {digit}: {precision:.4f}')

# Save weights
print("\nSaving model weights...")
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')
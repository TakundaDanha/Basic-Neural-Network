import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=5, hidden_size=10, output_size=3, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize all weights and biases to 1
        self.W1 = np.ones((self.input_size, self.hidden_size))
        self.b1 = np.ones((1, self.hidden_size))
        self.W2 = np.ones((self.hidden_size, self.output_size))
        self.b2 = np.ones((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        self.hidden_input = np.dot(X, self.W1) + self.b1
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.W2) + self.b2
        self.output = self.sigmoid(self.output_input)
        return self.output

    def sum_of_squares_loss(self, y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def backpropagate(self, X, y):
        output_error = self.output - y
        output_delta = output_error * self.sigmoid_derivative(self.output)
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        self.W2 -= self.learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.b2 -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.W1 -= self.learning_rate * np.dot(X.T, hidden_delta)
        self.b1 -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

# Read input from user in Jupyter Notebook
inputs = [float(input()) for _ in range(8)]

# Ensure exactly 8 values are entered
if len(inputs) != 8:
    raise ValueError("Expected exactly 8 values (5 inputs, 3 targets).")

# Convert to float
#inputs = [Float(input()) for _ in range(8)]
X_input = np.array(inputs[:5]).reshape(1, -1)  # First 5 values are inputs
y_target = np.array(inputs[5:]).reshape(1, -1)  # Last 3 values are targets

# Initialize and process neural network
nn = NeuralNetwork()
output_before = nn.feedforward(X_input)
loss_before = nn.sum_of_squares_loss(output_before, y_target)

# Perform one iteration of backpropagation
nn.backpropagate(X_input, y_target)
output_after = nn.feedforward(X_input)
loss_after = nn.sum_of_squares_loss(output_after, y_target)

# Print losses rounded to 4 decimal places
print(f"{loss_before:.4f}")
print(f"{loss_after:.4f}")

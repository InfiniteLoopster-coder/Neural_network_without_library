import numpy as np

# Activation functions and their derivatives

def sigmoid(x):
    """
    Computes the sigmoid of x.

    The sigmoid function is defined as 1 / (1 + exp(-x)), where exp is the
    exponential function. It maps any real-valued number into the (0, 1) range.

    Parameters:
    x : array_like
        Input value or array of values.

    Returns:
    array_like
        The sigmoid of the input value(s), with the same shape as x.
    """

    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Computes the derivative of the sigmoid function at x.

    Parameters:
    x : array_like
        Input value or array of values.

    Returns:
    array_like
        The derivative of the sigmoid function at the input value(s), with the same shape as x.
    """

    s = sigmoid(x)
    return s * (1.0 - s)


# Generate some dummy data for demonstration
# Let's say we have 4 examples with 2 features each and binary targets.

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0],
              [1],
              [1],
              [0]])

# Set the number of neurons in hidden layer
n_feature = X.shape[1]
n_hidden = 3
n_output = 1


# Hyperparameters

learning_rate = 0.1
num_iteration = 10000


# Initialize weight and biases

np.random.seed(42)
weights_input_hidden = np.random.rand(n_feature, n_hidden) - 0.5
bias_hidden = np.zeros((1, n_hidden))

weights_hidden_output = np.random.rand(n_hidden, n_output) - 0.5
bias_output = np.zeros((1, n_output))


# Training loop

for i in range(num_iteration):
    z_hidden = np.dot(X, weights_input_hidden) + bias_hidden
    a_hidden = sigmoid(z_hidden)

    z_output = np.dot(a_hidden, weights_hidden_output) + bias_output
    a_output = sigmoid(z_output)


    # Compute loss (mean squared error)
    loss = np.mean((Y - a_output)**2)

    # Backpropagation
    # Output layer error and gradient
    error_output = a_output - Y
    delta_output = error_output * sigmoid_derivative(z_output)
    
    # Hidden layer error and gradient
    error_hidden = np.dot(delta_output, weights_hidden_output.T)
    delta_hidden = error_hidden * sigmoid_derivative(z_hidden)
    
    # Gradients for weights and biases
    grad_weights_hidden_output = np.dot(a_hidden.T, delta_output)
    grad_bias_output = np.sum(delta_output, axis=0, keepdims=True)
    
    grad_weights_input_hidden = np.dot(X.T, delta_hidden)
    grad_bias_hidden = np.sum(delta_hidden, axis=0, keepdims=True)
    
    # Update weights and biases
    weights_hidden_output -= learning_rate * grad_weights_hidden_output
    bias_output -= learning_rate * grad_bias_output
    weights_input_hidden -= learning_rate * grad_weights_input_hidden
    bias_hidden -= learning_rate * grad_bias_hidden

    # Optionally print loss every 1000 iterations to monitor training progress
    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {loss}")

# After training, check the network's output
print("Final output after training:")
print(a_output)




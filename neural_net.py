# Created by Ethan Edwards on 12/8/2024

# Imports
import numpy as np

# Main class
class neural_net:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def one_hot_encode(self, y, num_classes):
        return np.eye(num_classes)[y]

    def forward_pass(self, X):
        # Input to hidden layer
        self.z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a_hidden = self.sigmoid(self.z_hidden)

        # Hidden to output layer
        self.z_output = np.dot(self.a_hidden, self.weights_hidden_output) + self.bias_output
        self.a_output = self.softmax(self.z_output)

        return self.a_output

    def backward_pass(self, X, y_true):
        m = X.shape[0]

        # Calculate output layer error
        error_output = self.a_output - y_true

        # Calculate hidden layer error
        error_hidden = np.dot(error_output, self.weights_hidden_output.T) * self.sigmoid_derivative(self.a_hidden)

        # Gradient for weights and biases
        grad_weights_hidden_output = np.dot(self.a_hidden.T, error_output) / m
        grad_bias_output = np.sum(error_output, axis=0, keepdims=True) / m

        grad_weights_input_hidden = np.dot(X.T, error_hidden) / m
        grad_bias_hidden = np.sum(error_hidden, axis=0, keepdims=True) / m

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * grad_weights_hidden_output
        self.bias_output -= self.learning_rate * grad_bias_output
        self.weights_input_hidden -= self.learning_rate * grad_weights_input_hidden
        self.bias_hidden -= self.learning_rate * grad_bias_hidden

    def train(self, x_train, y_train, x_val, y_val, num_epochs=1000, target_accuracy=0.95):
        y_train_encoded = self.one_hot_encode(y_train, self.output_size)
        y_val_encoded = self.one_hot_encode(y_val, self.output_size)

        best_val_accuracy = 0  # To track the best validation accuracy
        best_weights_input_hidden = self.weights_input_hidden.copy()
        best_bias_hidden = self.bias_hidden.copy()
        best_weights_hidden_output = self.weights_hidden_output.copy()
        best_bias_output = self.bias_output.copy()

        for epoch in range(num_epochs):
            # Forward pass
            predictions_train = self.forward_pass(x_train)

            # Backward pass
            self.backward_pass(x_train, y_train_encoded)

            # Validation phase
            predictions_val = self.forward_pass(x_val)
            val_accuracy = np.mean(np.argmax(predictions_val, axis=1) == y_val)

            # Update the best accuracy and store the best weights if necessary
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # Save the weights and biases for the best accuracy
                best_weights_input_hidden = self.weights_input_hidden.copy()
                best_bias_hidden = self.bias_hidden.copy()
                best_weights_hidden_output = self.weights_hidden_output.copy()
                best_bias_output = self.bias_output.copy()

            print(f"\rEpoch: {epoch + 1}/{num_epochs} | Val Accuracy: {val_accuracy:.4f} | Val Best: {best_val_accuracy:.4f}", end="")

            # Check if target accuracy is achieved
            if val_accuracy >= target_accuracy:
                print("Target accuracy reached.")
                break

        # Restore the best weights
        self.weights_input_hidden = best_weights_input_hidden
        self.bias_hidden = best_bias_hidden
        self.weights_hidden_output = best_weights_hidden_output
        self.bias_output = best_bias_output

        print("\n")

    def predict(self, X):
        predictions = self.forward_pass(X)
        return np.argmax(predictions, axis=1)

    def validate(self, x_test, y_test):
        # Flatten x_test if it is in the 2D image form
        x_test = x_test.reshape(x_test.shape[0], -1)

        # Get predictions
        predictions = self.predict(x_test)

        # Overall accuracy
        overall_accuracy = np.mean(predictions == y_test)
        print(f"Overall accuracy: {overall_accuracy * 100:.4f}%")

        # Individual label accuracy
        unique_labels = np.unique(y_test)
        for label in unique_labels:
            label_accuracy = np.mean(predictions[y_test == label] == label)
            print(f"Accuracy for label {label}: {label_accuracy * 100:.4f}%")

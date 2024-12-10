# Created by Ethan Edwards on 12/8/2024

# Imports
import numpy as np

# Main class
class perceptron:
    def __init__(self, num_classes, num_features, lr):
        self.weights = np.random.uniform(-0.1, 0.1, (num_classes, num_features))
        self.best = np.copy(self.weights)  # Best weights (for validation)
        self.learning_rate = lr

    def predict(self, features, training=False, w=None):
        """
        Predict the class label
        """
        if training:
            w = self.weights  # Use the training weights
        elif w is None:
            w = self.best  # Use the best weights if not provided

        # Compute scores for all classes
        scores = np.dot(w, features)

        # Return the class with the highest score
        predicted_label = np.argmax(scores)

        return predicted_label

    def train(self, train_data, val_data, epochs, num_classes):
        """
        Train the model using a multiclass perceptron algorithm
        """
        # Initialize weights for each class with small random values
        self.weights = np.random.uniform(-0.1, 0.1, (num_classes, len(train_data[0]) - 1))
        best_accuracy = 0

        # Create a mapping from class labels to indices
        unique_labels = sorted(set(sample[-1] for sample in train_data))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        for epoch in range(epochs):
            for sample in train_data:
                features = np.array(sample[:-1])  # Feature vector
                true_label = label_to_index[sample[-1]]  # Normalize true label

                # Predict the class with the highest score
                scores = np.dot(self.weights, features)
                predicted_label = np.argmax(scores)

                if predicted_label != true_label:
                    # Update weights
                    self.weights[true_label] += self.learning_rate * features
                    self.weights[predicted_label] -= self.learning_rate * features

            # Evaluate accuracy on validation data
            correct = 0
            for sample in val_data:
                features = np.array(sample[:-1])
                true_label = label_to_index[sample[-1]]  # Normalize true label

                scores = np.dot(self.weights, features)
                predicted_label = np.argmax(scores)

                if predicted_label == true_label:
                    correct += 1

            accuracy = correct / len(val_data)
            print(f"\rEpoch: {epoch + 1}/{epochs} | Val Accuracy: {accuracy:.4f} | Val Best: {best_accuracy:.4f}", end="")

            # Track the best weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best = self.weights.copy()

        print("\n")

    def eval(self, test_data):
        """
        Evaluate the model on test data and display overall and individual label accuracies.
        """
        correct = 0
        # Initialize dictionaries to track correct predictions and total counts per label
        label_correct = {label: 0 for label in set(sample[-1] for sample in test_data)}
        label_total = {label: 0 for label in set(sample[-1] for sample in test_data)}

        # Normalize labels (same mapping as during training)
        unique_labels = sorted(set(sample[-1] for sample in test_data))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        index_to_label = {idx: label for label, idx in label_to_index.items()}

        for sample in test_data:
            features = np.array(sample[:-1])  # Extract features
            true_label = label_to_index[sample[-1]]  # Normalize true label
            prediction_index = self.predict(features)  # Get predicted label index
            prediction = index_to_label[prediction_index]  # Convert index back to label

            # Update overall accuracy
            if prediction == sample[-1]:
                correct += 1
                label_correct[sample[-1]] += 1  # Increment correct count for this label
            label_total[sample[-1]] += 1  # Increment total count for this label

        # Compute overall accuracy
        accuracy = correct / len(test_data)
        print(f"Overall Test Accuracy: {accuracy * 100:.4f}%")

        # Compute and display individual label accuracies
        print("\nIndividual Label Accuracy:")
        for label in label_total:
            if label_total[label] > 0:
                label_accuracy = label_correct[label] / label_total[label]
                print(f"Label {int(label)}: {label_accuracy * 100:.4f}%")
            else:
                print(f"Label {label}: No samples")

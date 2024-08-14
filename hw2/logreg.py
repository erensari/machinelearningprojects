from typing import List
import random
import math

class LogisticRegression:
    def __init__(self, learning_rate: float, epochs: int, batch_size: int):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = 10 
        self.weights = None
        self.biases = None

   
    def fit(self, X: List[List[float]], y: List[int]):
        numSamples = len(X)
        numFeatures = len(X[0])

        self._initialize_weights(numFeatures)

        for _ in range(self.epochs):
            indices = list(range(numSamples))
            random.shuffle(indices)

            for i in range(0, numSamples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = [X[idx] for idx in batch_indices]
                y_batch = [y[idx] for idx in batch_indices]

                gradients_w, gradients_b = self._compute_gradients(X_batch, y_batch)

                self._update_weights_and_biases(gradients_w, gradients_b, len(X_batch))


    def predict_proba(self, inputs):
        probabilities = []
        for sample in inputs:
            logits = [
                sum(sample[feature_index] * self.weights[feature_index][class_index] for feature_index in range(len(sample))) 
                + self.biases[class_index] 
                for class_index in range(self.num_classes)
            ]
            probabilities.append(self.softmax(logits))
        return probabilities

    def predict(self, inputs):
        probabilities = self.predict_proba(inputs)
        predictions = [proba.index(max(proba)) for proba in probabilities]
        return predictions

    
    
    def softmax(self, logits):
        exp_logits = [math.exp(logit - max(logits)) for logit in logits]
        sum_exp_logits = sum(exp_logits)
        softmax_output = [exp_logit / sum_exp_logits for exp_logit in exp_logits]
        return softmax_output

    def cross_entropy_loss(self, true_labels, predicted_probs):
        loss = -sum(true_labels[i] * math.log(predicted_probs[i]) for i in range(len(true_labels)))
        return loss
    
    
    def _initialize_weights(self, numFeatures: int):
        self.weights = [[random.uniform(-0.01, 0.01) for _ in range(self.num_classes)] for _ in range(numFeatures)]
        self.biases = [random.uniform(-0.01, 0.01) for _ in range(self.num_classes)]


    def _compute_gradients(self, X_batch: List[List[float]], y_batch: List[int]):
        numFeatures = len(X_batch[0])
        gradients_w = [[0.0] * self.num_classes for _ in range(numFeatures)]
        gradients_b = [0.0] * self.num_classes

        for x, label in zip(X_batch, y_batch):
            predictions = self.predict_proba([x])[0]
            y_true = [1 if i == label else 0 for i in range(self.num_classes)]
            error = [predictions[i] - y_true[i] for i in range(self.num_classes)]

            for j in range(numFeatures):
                for k in range(self.num_classes):
                    gradients_w[j][k] += error[k] * x[j]

            for k in range(self.num_classes):
                gradients_b[k] += error[k]

        return gradients_w, gradients_b


    def _update_weights_and_biases(self, gradients_w: List[List[float]], gradients_b: List[float], batch_size: int):
        numFeatures = len(gradients_w)

        for j in range(numFeatures):
            for k in range(self.num_classes):
                self.weights[j][k] -= self.learning_rate * (gradients_w[j][k] / batch_size)

        for k in range(self.num_classes):
            self.biases[k] -= self.learning_rate * (gradients_b[k] / batch_size)


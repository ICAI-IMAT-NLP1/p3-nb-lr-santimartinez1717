import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = estimate_class_priors(labels)
        self.vocab_size = None # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = estimate_conditional_probabilities(features, labels, delta)
        

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # Count number of samples for each output class and divide by total of samples
        labels = labels.flatten()

        # Get the total number of examples
        total_samples = labels.size(0)

        # Count occurrences of each class in the labels 
        class_counts = torch.bincount(labels)

        # Calculate class priors: P(class) = count(class) / total_samples
        class_priors = {class_label.item(): class_count.float() / total_samples
                        for class_label, class_count in enumerate(class_counts)}

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """

        class_word_counts: Dict[int, torch.Tensor] = {}
        total_word_counts: Dict[int, torch.Tensor] = {}
        smoothed_conditional_probabilities: Dict[int, torch.Tensor]  = {} 
        n_classes = labels.size(0)

        for label in labels.unique():

            mask = (label == labels)
            class_features = features[mask]
            label_bow = sum(class_features, dim=0)
            label_total_words = sum(label_bow, dim=0)

            class_word_counts[label] = label_bow
            total_word_counts[label] = label_total_words

        for label, bows in class_word_counts.items():

            total_count = total_word_counts[class_label]
            smoothed_conditional_probabilities[label] = (bows + delta) / (total_count + delta*n_classes)

        

        return smoothed_conditional_probabilities

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # Calculate posterior based on priors and conditional probabilities of the words
        log_posteriors: torch.Tensor = None


        # Convert the dictionary of conditional probabilities to a tensor
        log_conditional_probs = torch.stack([torch.log(word_probs) for word_probs in self.conditional_probabilities.values()], dim=0)

        # Convert class priors to a tensor of log(prior)
        log_class_priors = torch.log(torch.tensor(list(self.class_priors.values())))

        # Calculate the log posteriors for all classes using efficient tensor operations
        log_posteriors = torch.sum(log_conditional_probs * features, dim=1) + log_class_priors

        return log_posteriors

        

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        #  Calculate log posteriors and obtain the class of maximum likelihood 
        log_posteriors = self.estimate_class_posteriors(feature)
        
        # Find the class with the highest log posterior
        pred = torch.argmax(log_posteriors).item()  # Get the index of the highest value
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # Calculate log posteriors and transform them to probabilities (softmax)
        log_posteriors = self.estimate_class_posteriors(feature)
        
        # Convert log posteriors to probabilities using softmax
        probs = torch.softmax(log_posteriors, dim=0)  # softmax to get a valid probability distribution
        return probs

"""
ClassifAIer Module

This module provides the ClassifAIer class for text classification tasks
using embeddings generated from large language models. It integrates with the 
Langchain embedding library to leverage various embedding models, enabling users 
to train classifiers from text data and make predictions.

Key Features:
- Support for various sklearn classifiers.
- Compatibility with Langchain embeddings for generating text embeddings.
- Functions to fit models, make predictions, save, and load classifiers.

Usage:
    To use this module, import the ClassifAIer class and initialize it 
    with an appropriate embedding provider. Train the model using the fit 
    method and make predictions using the predict method.
"""

import pickle
from typing import Any, List
from langchain_core.embeddings import Embeddings
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


class ClassifAIer:
    """A classifier that utilizes large language model embeddings for
    text classification tasks.

    This class integrates with the Langchain embedding library to
    obtain embeddings from large language models and uses
    various sklearn classifiers for supervised training.
    It supports loading, saving, and predicting on text data.

    Attributes:
        embeddings (Embeddings): The embedding provider instance used for
            generating text embeddings (Langchain embeddings).
        label_encoder (LabelEncoder): Encoder for transforming labels
            into a format suitable for training.
        classifier (Any): The classifier instance used for making predictions.
    """

    def __init__(
        self, embedding_provider: Embeddings, classifier: ClassifierMixin = None
    ):
        self.embeddings = embedding_provider
        self.label_encoder = LabelEncoder()
        self.classifier = (
            classifier
            if classifier is not None
            else KNeighborsClassifier(n_neighbors=3)
        )  # Default to KNeighborsClassifier

    def get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a given text.

        Args:
            text (str): The input text for which the embedding is to be obtained.

        Returns:
            List[float]: A list of floats representing the embedding of the input text.
        """
        return self.embeddings.embed_query(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings for a given text list.

        Args:
            texts (List[str]): The input texts for which the embeddings is to be obtained.

        Returns:
            List[List[float]]: A list of list of floats representing the embeddings
                of the input texts.
        """
        return self.embeddings.embed_documents(texts)

    def fit(
        self,
        texts: List[str],
        labels: List[Any],
        given_embeddings: List[List[float]] = None,
    ) -> None:
        """Train the classifier on the provided texts and labels.

        Args:
            texts (List[str]): A collection of input texts used for training the classifier.
            labels (List[Any]): The corresponding labels for the input texts,
                used to supervise the training process.
            given_embeddings (List[List[float]], optional): Pre-computed embeddings for the input texts.
                If provided, these will be used instead of generating embeddings from `texts`.


        Returns:
            None: This method does not return any value.

        Raises:
            ValueError: If the number of texts does not match the number of labels.
        """
        if given_embeddings is None and len(texts) != len(labels):
            raise ValueError("The number of texts and labels must be the same.")
        if len(texts) != len(labels) or (
            given_embeddings is not None and len(texts) != len(given_embeddings)
        ):
            raise ValueError(
                "The number of texts, labels, and given embeddings must be the same."
            )

        embeddings = np.array(
            given_embeddings
            if given_embeddings is not None
            else self.get_embeddings(texts)
        )
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.classifier.fit(embeddings, encoded_labels)
        print("The model has been successfully trained.")

    def predict(self, text: str) -> Any:
        """Predict the label for a given input text.

        Args:
            text (str): The input text for which the prediction is to be made.

        Returns:
            Any: The predicted label for the input text.
        """
        embedding = self.get_embedding(text)
        prediction = self.classifier.predict([embedding])
        return self.label_encoder.inverse_transform(prediction)[0]

    def predict_all(self, texts: List[str]) -> List[Any]:
        """Predict the labels for a list of input texts.

        Args:
            texts (List[str]): A list of input texts for which predictions are to be made.

        Returns:
            List[Any]: A list of predicted labels for the input texts.
        """
        embeddings = np.array(self.get_embeddings(texts))
        predictions = self.classifier.predict(embeddings)
        return self.label_encoder.inverse_transform(predictions).tolist()

    def save(self, filename: str) -> None:
        """Save the trained classifier to a file.

        Args:
            filename (str): The name of the file where the classifier
                will be saved. It can be a string representing the
                file path.

        Returns:
            None: This method does not return any value.

        Raises:
            IOError: If there is an error in saving the file.
        """
        model_info = {
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
        }
        with open(filename, "wb") as file:
            pickle.dump(model_info, file)
        print(f"Model saved to file '{filename}'.")

    @staticmethod
    def load(filename: str, embedding_provider: Embeddings) -> "ClassifAIer":
        """Load a trained classifier from a file.

        Args:
            filename (str): The name of the file from which the classifier
                will be loaded. It can be a string representing the
                file path.
            embedding_provider (Embeddings): The embedding provider to be used
                for the classifier.

        Returns:
            ClassifAIer: An instance of ClassifAIer that
                has been loaded from the specified file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If there is an error in loading the file.
        """
        with open(filename, "rb") as file:
            model_info = pickle.load(file)

        classifier_instance = ClassifAIer(
            embedding_provider=embedding_provider, classifier=model_info["classifier"]
        )
        classifier_instance.label_encoder = model_info["label_encoder"]

        print(f"Model loaded from file '{filename}'.")
        return classifier_instance

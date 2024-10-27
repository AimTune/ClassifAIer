# ClassifAIer

ClassifAIer is a Python library that combines **scikit-learn** classifiers with **LangChain** embedding libraries, enabling seamless text classification using embeddings from large language models. This library offers a user-friendly interface, allowing you to classify text data in a human-like manner.

## Features

- **Embedding Support**: Ability to work with embeddings from large language models like OpenAI and compatible with embeddings supported by LangChain.
- **Parametric Classifiers**: Compatibility with a wide range of classifiers from `scikit-learn` (e.g., `RandomForestClassifier`, `KNeighborsClassifier`, etc.).
- **Easy to Use**: Simplifies text classification tasks with a user-friendly API.
- **Save and Load**: Allows you to save and reload trained models for reuse.

## Requirements

To use this library, you need Python 3.7 or higher. The required packages will be automatically installed when you install this library.

- scikit-learn
- langchain-core
- langchain
- numpy

## Installation

You can install the required libraries using the following command:

```bash
pip install ClassifAIer
```

## Usage

```python
from classifaier import ClassifAIer
from langchain.embeddings import OpenAIEmbeddings

# Initialize the embedding provider
embedding_provider = OpenAIEmbeddings(api_key='YOUR_API_KEY')

random_forest_classifier_params = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
}

random_forest_classifier = RandomForestClassifier(**random_forest_classifier_params)

# Create a classifier instance
classifier = ClassifAIer(embedding_provider=embedding_provider, classifier=random_forest_classifier)

# Prepare your data
texts = ["This is a positive review.", "This is a negative review."]
labels = ["positive", "negative"]

# Train the model
classifier.fit(texts, labels)

# Make predictions
positive_prediction = classifier.predict("I love this product!")
print(positive_prediction)  # Output: 'positive'

negative_prediction = classifier.predict("I hate this service.")
print(negative_prediction)  # Output: negative

# Save the model
classifier.save("my_classifier.pkl")

# Load the model
loaded_classifier = ClassifAIer.load("my_classifier.pkl", embedding_provider)
```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
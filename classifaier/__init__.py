"""
ClassifAIer: A Python library for text classification using embeddings 
from large language models with scikit-learn classifiers.

This package combines the power of LangChain embedding libraries with 
various classifiers from scikit-learn to enable seamless and human-like 
text classification.

"""

# Import the main classifier class
from .classifaier import ClassifAIer

__all__ = [
    "ClassifAIer",
]

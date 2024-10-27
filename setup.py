"""
Setup script for ClassifAIer package.

This script uses setuptools to package and distribute the ClassifAIer library,
which provides tools for text classification using embeddings from large language models.
It integrates with the Langchain embedding library and supports various sklearn classifiers.

Usage:
    To install this package, run:
        python setup.py install

    Alternatively, you can use pip:
        pip install .
"""

from setuptools import setup, find_packages

setup(
    name="ClassifAIer",  # Updated project name
    version="0.1.4",
    author="Hamza Agar",
    author_email="hamzaagareng@gmail.com",
    description="A library for text classification using LangChain embeddings and "
    "scikit-learn classifiers.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AimTune/ClassifAIer",  # Update with your GitHub link
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "langchain-core",
        "langchain",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

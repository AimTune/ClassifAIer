import pytest
from classifaier import ClassifAIer
from langchain_openai import OpenAIEmbeddings
from sklearn.neighbors import KNeighborsClassifier
from datasets import load_dataset
from sklearn.metrics import f1_score
#from dotenv import load_dotenv

#load_dotenv()

def test_classifier_initialization():
    embeddings = OpenAIEmbeddings()
    classifier = ClassifAIer(embedding_provider=embeddings)
    assert classifier is not None
    assert isinstance(classifier.classifier, KNeighborsClassifier)

def test_fit_and_predict():
    embeddings = OpenAIEmbeddings()
    classifier = ClassifAIer(embedding_provider=embeddings)
    
    texts = ["This is a test", "This is another test", "This is a third test", "This is a fourth test"]
    labels = ["test", "test2", "test3", "test4"]
    
    classifier.fit(texts, labels)
    prediction = classifier.predict("This is a test")
    assert prediction in labels

def test_predict_all():
    embeddings = OpenAIEmbeddings()
    classifier = ClassifAIer(embedding_provider=embeddings)
    
    texts = ["This is a test", "This is another test", "This is a third test", "This is a fourth test"]
    labels = ["test", "test2", "test3", "test4"]
    
    classifier.fit(texts, labels)
    predictions = classifier.predict_all(["This is a test", "This is another test"])
    assert all(pred in labels for pred in predictions)

def test_save_and_load():
    embeddings = OpenAIEmbeddings()
    classifier = ClassifAIer(embedding_provider=embeddings)
    
    texts = ["This is a test", "This is another test", "This is a third test", "This is a fourth test"]
    labels = ["test", "test2", "test3", "test4"]
    
    classifier.fit(texts, labels)
    classifier.save("test_model.pkl")
    
    loaded_classifier = ClassifAIer.load("test_model.pkl", embeddings)
    prediction = loaded_classifier.predict("This is a test")
    assert prediction in labels

def test_load_from_dataset():
    embeddings = OpenAIEmbeddings()
    classifier = ClassifAIer(embedding_provider=embeddings)

    dataset = load_dataset("sh0416/ag_news", split="train[:100]")
    descriptions = [sample["description"] for sample in dataset]
    labels = [sample["label"] for sample in dataset]

    test_dataset = load_dataset("sh0416/ag_news", split="test[:20]")
    test_descriptions = [sample["description"] for sample in test_dataset]
    test_labels = [sample["label"] for sample in test_dataset]
    
    classifier.fit(descriptions, labels)
    predictions = classifier.predict_all(test_descriptions)
    
    f1 = f1_score(test_labels, predictions, average='weighted')
    print(f"F1 Score: {f1:.2f}")
    assert f1 >= 0.5, f"F1 score {f1:.2f} is below 0.5"


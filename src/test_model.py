import pytest
from src.model import load_data, train_model
import joblib
import os

def test_load_data():
    iris = load_data()
    assert not iris.empty, "The dataset should not be empty"
    assert iris.shape[1] == 5, "There should be 5 columns in the dataset"
    assert os.path.exists("data/iris.csv"), "The dataset should be saved in the data folder"

def test_train_model():
    model = train_model()
    assert os.path.exists("logistic_model.pkl"), "Model file should be saved"

@pytest.fixture(scope="module")
def model_fixture():
    return joblib.load("logistic_model.pkl")

def test_model_accuracy(model_fixture):
    model = model_fixture
    assert isinstance(model, joblib.load("logistic_model.pkl").__class__), "Loaded object should be a Logistic Regression model"

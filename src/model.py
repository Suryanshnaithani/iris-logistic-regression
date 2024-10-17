import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def load_data():
    """Loads and saves the Iris dataset into the data folder"""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    iris = pd.read_csv(url, header=None, names=col_names)
    iris['species'] = iris['species'].astype('category').cat.codes  # Encode labels

    # Create the data directory if it doesn't exist and save the dataset
    os.makedirs('data', exist_ok=True)
    iris.to_csv('data/iris.csv', index=False)

    return iris

def train_model():
    """Train a logistic regression model"""
    iris = load_data()
    X = iris.drop('species', axis=1)
    y = iris['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.2f}")

    # Save the model
    joblib.dump(model, 'logistic_model.pkl')

if __name__ == "__main__":
    train_model()

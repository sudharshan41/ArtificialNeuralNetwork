import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and preprocess the data
# input: filepath: str (path to the CSV file)
# output: tuple of X (features), y (target)
def load_and_preprocess_data(filepath):
    # Load the data from the file
    data = pd.read_csv(filepath)
    
    # Drop the "GarbageValues" column if it exists
    if 'GarbageValues' in data.columns:
        data = data.drop('GarbageValues', axis=1)
    
    # Remove rows with missing values
    data = data.dropna()

    # Assuming the last column is the target variable and the rest are features
    X = data.drop('target', axis=1).values
    y = data['target'].values
    
    # Label encoding the target variable if it's categorical
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y

# Split the data into training and testing sets and standardize the features
# input: 1) X: ndarray (features)
#        2) y: ndarray (target)
# output: tuple of X_train, X_test, y_train, y_test
def split_and_standardize(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Create and train 2 MLP classifiers with different parameters
# input:  1) X_train: ndarray
#         2) y_train: ndarray
# output: tuple of models (model1, model2)
def create_model(X_train, y_train):
    # Model 1: Three hidden layers with ReLU activation
    model1 = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', max_iter=100, solver='adam', random_state=42)
    
    # Model 2: Three hidden layers with Tanh activation
    model2 = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='tanh', max_iter=150, solver='sgd', random_state=1)
    
    # Train both models
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    
    return model1, model2

# Predict and evaluate the model
# input: 1) model: MLPClassifier after training
#        2) X_test: ndarray
#        3) y_test: ndarray
# output: tuple - accuracy, precision, recall, fscore, confusion_matrix
def predict_and_evaluate(model, X_test, y_test):
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    fscore = f1_score(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, fscore, cm
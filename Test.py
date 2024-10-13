import sys
import importlib
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--ID", required=True, help="Name of the module to import")
parser.add_argument("--filepath", required=True, help="Path to the CSV file")
args = parser.parse_args()
subname = args.ID
filepath = args.filepath

# Load the module specified by the ID argument
try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print("Failed to import module. Ensure your module file name matches the --ID argument.")
    print(f"Error: {e}")
    sys.exit()

# Extract functions from the module
try:
    split_and_standardize = mymodule.split_and_standardize
    create_model = mymodule.create_model
    predict_and_evaluate = mymodule.predict_and_evaluate
except AttributeError as e:
    print(f"Error: The module does not contain the required functions. Check if the functions are named correctly.")
    print(f"Error: {e}")
    sys.exit()

# Load and preprocess the data
def load_and_preprocess_data(filepath):
    try:
        # Load the dataset
        data = pd.read_csv(filepath)
        
        # Remove garbage values column if it exists
        if 'GarbageValues' in data.columns:
            data = data.drop(columns=['GarbageValues'])
        
        # Drop rows with missing values
        data = data.dropna()
        
        # Check if the target column is 'Outcome'
        if 'Outcome' in data.columns:
            # Split the data into features (X) and target (y)
            X = data.drop('Outcome', axis=1)
            y = data['Outcome']
        else:
            raise ValueError("Target column 'Outcome' not found in the dataset.")
        
        # Ensure y is a 1D array
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.values.ravel()
        
        return X, y
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        sys.exit()

def test_case():
    X, y = load_and_preprocess_data(filepath)
    model1 = model2 = None
    X_train = X_test = y_train = y_test = None
    
    try:
        X_train, X_test, y_train, y_test = split_and_standardize(X, y)
        if (X_train.shape[0] + X_test.shape[0] == X.shape[0] and 
            X.shape[0] == (y_train.shape[0] + y_test.shape[0]) and
            np.allclose(np.mean(X_train), 0, atol=1e-1) and 
            np.allclose(np.mean(X_test), 0, atol=1e-1)):
            print("Test Case 1 for the function split_and_standardize PASSED")
        else:
            print("Test Case 1 for the function split_and_standardize FAILED")
    except Exception as e:
        print(f"Error in split_and_standardize: {e}")
        print("Test Case 1 for the function split_and_standardize FAILED [ERROR]")

    try:
        model1, model2 = create_model(X_train, y_train)
        if (len(model1.get_params()['hidden_layer_sizes']) == 3 and 
            len(model2.get_params()['hidden_layer_sizes']) == 3 and 
            model1.get_params()['activation'] != model2.get_params()['activation']):
            print("Test Case 2 for the function create_model PASSED")
        else:
            print("Test Case 2 for the function create_model FAILED")
    except Exception as e:
        print(f"Error in create_model: {e}")
        print("Test Case 2 for the function create_model FAILED [ERROR]")

    try:
        accuracy, precision, recall, fscore, conf_matrix = predict_and_evaluate(model1, X_test, y_test)
        if (accuracy >= 0.70 and precision >= 0.60 and recall >= 0.70 and fscore >= 0.70):
            print("Test Case 3 for the function predict_and_evaluate PASSED")
        else:
            print("Test Case 3 for the function predict_and_evaluate FAILED")
    except Exception as e:
        print(f"Error in predict_and_evaluate for model1: {e}")
        print("Test Case 3 for the function predict_and_evaluate FAILED [ERROR]")

    try:
        accuracy, precision, recall, fscore, conf_matrix = predict_and_evaluate(model2, X_test, y_test)
        if (0.5 <= accuracy < 1.0 and 
            0.5 <= precision < 1.0 and 
            0.5 <= recall < 1.0 and 
            0.5 < fscore < 1.0):
            print("Test Case 4 for the function predict_and_evaluate PASSED")
        else:
            print("Test Case 4 for the function predict_and_evaluate FAILED")
    except Exception as e:
        print(f"Error in predict_and_evaluate for model2: {e}")
        print("Test Case 4 for the function predict_and_evaluate FAILED [ERROR]")

if __name__ == "__main__":
    test_case()

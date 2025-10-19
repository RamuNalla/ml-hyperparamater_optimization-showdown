import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
import json
import os

def load_wine_data():
    """Load wine quality dataset from UCI repository"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(url, sep=';')
        return df
    except:
        print("Error loading data from URL. Using local file if available.")
        df = pd.read_csv('data/winequality-red.csv', sep=';')
        return df
    
def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare data for training"""
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Convert to binary classification for simplicity (good wine >= 6)
    y = (y >= 6).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Convert predictions to plain Python list so they're JSON serializable
    try:
        preds = y_pred.tolist()
    except AttributeError:
        preds = list(y_pred)

    return {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'predictions': preds
    }

def save_results(method_name, best_params, metrics, time_taken, n_iterations):
    """Save optimization results"""
    # Custom JSON encoder to handle numpy types
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    results = {
        'method': method_name,
        'best_params': best_params,
        'metrics': metrics,
        'time_taken': time_taken,
        'n_iterations': n_iterations,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    os.makedirs('results', exist_ok=True)
    # Sanitize filename
    sanitized_method_name = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filename = f'results/{sanitized_method_name}_results.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, cls=NpEncoder)
    
    print(f"Results saved to {filename}")
    return results

def timer_decorator(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper
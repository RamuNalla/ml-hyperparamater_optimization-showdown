from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import time
import sys
sys.path.append('..')
from utils import evaluate_model, save_results

def run_random_search(X_train, y_train, X_test, y_test, model_type='random_forest', n_iter=50):
    """Run Random Search optimization"""
    print(f"\n{'='*60}")
    print(f"Running Random Search with {model_type.upper()}")
    print(f"{'='*60}\n")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_distributions = {
            'n_estimators': randint(50, 300),
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }
    else:  # xgboost
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        param_distributions = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5)
        }
    
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    best_model = random_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)
    
    print(f"\nBest Parameters: {random_search.best_params_}")
    print(f"Best CV Score: {random_search.best_score_:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")
    print(f"Time Taken: {time_taken:.2f} seconds")
    print(f"Total Iterations: {n_iter}")
    
    results = save_results(
        method_name=f'Random Search ({model_type})',
        best_params=random_search.best_params_,
        metrics=metrics,
        time_taken=time_taken,
        n_iterations=n_iter
    )
    
    return best_model, results

if __name__ == "__main__":
    from utils import load_wine_data, prepare_data
    
    df = load_wine_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    run_random_search(X_train, y_train, X_test, y_test, model_type='random_forest')

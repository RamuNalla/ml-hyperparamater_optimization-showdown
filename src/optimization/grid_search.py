from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
import sys
sys.path.append('..')
from utils import evaluate_model, save_results

def run_grid_search(X_train, y_train, X_test, y_test, model_type='random_forest'):
    """Run Grid Search optimization"""
    print(f"\n{'='*60}")
    print(f"Running Grid Search with {model_type.upper()}")
    print(f"{'='*60}\n")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:  # xgboost
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 0.8, 0.9]
        }

    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,                                   # split the data into 5 parts, train on 4 parts, test on 1 part and repeat 5 times
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")
    print(f"Time Taken: {time_taken:.2f} seconds")
    print(f"Total Iterations: {len(grid_search.cv_results_['params'])}")
    
    results = save_results(
        method_name=f'Grid Search ({model_type})',
        best_params=grid_search.best_params_,
        metrics=metrics,
        time_taken=time_taken,
        n_iterations=len(grid_search.cv_results_['params'])
    )
    
    return best_model, results

if __name__ == "__main__":
    from utils import load_wine_data, prepare_data
    
    df = load_wine_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    run_grid_search(X_train, y_train, X_test, y_test, model_type='random_forest')
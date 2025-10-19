from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
import sys
sys.path.append('..')
from utils import evaluate_model, save_results

def run_bayesian_optimization(X_train, y_train, X_test, y_test, model_type='random_forest', n_iter=50):
    """Run Bayesian Optimization using scikit-optimize"""
    print(f"\n{'='*60}")
    print(f"Running Bayesian Optimization with {model_type.upper()}")
    print(f"{'='*60}\n")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        search_spaces = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(5, 30),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None])
        }
    else:  # xgboost
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        search_spaces = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'gamma': Real(0, 0.5)
        }

        start_time = time.time()
    
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    bayes_search.fit(X_train, y_train)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    best_model = bayes_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)
    
    print(f"\nBest Parameters: {bayes_search.best_params_}")
    print(f"Best CV Score: {bayes_search.best_score_:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")
    print(f"Time Taken: {time_taken:.2f} seconds")
    print(f"Total Iterations: {n_iter}")
    
    results = save_results(
        method_name=f'Bayesian Optimization ({model_type})',
        best_params=bayes_search.best_params_,
        metrics=metrics,
        time_taken=time_taken,
        n_iterations=n_iter
    )
    
    return best_model, results

if __name__ == "__main__":
    from utils import load_wine_data, prepare_data
    
    df = load_wine_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    run_bayesian_optimization(X_train, y_train, X_test, y_test, model_type='random_forest')
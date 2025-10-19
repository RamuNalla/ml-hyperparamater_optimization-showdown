import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import time
import sys
sys.path.append('..')
from utils import evaluate_model, save_results

def run_optuna_optimization(X_train, y_train, X_test, y_test, model_type='random_forest', n_trials=50):
    """Run Optuna optimization"""
    print(f"\n{'='*60}")
    print(f"Running Optuna Optimization with {model_type.upper()}")
    print(f"{'='*60}\n")
    
    def objective(trial):                       # Single trial or one complete experiment
        if model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
        else:  # xgboost
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            model = XGBClassifier(**params)
        
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
        return score.mean()
    
    start_time = time.time()
    
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    best_params = study.best_params
    
    if model_type == 'random_forest':
        best_model = RandomForestClassifier(**best_params, random_state=42)
    else:
        best_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
    
    best_model.fit(X_train, y_train)
    metrics = evaluate_model(best_model, X_test, y_test)
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV Score: {study.best_value:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")
    print(f"Time Taken: {time_taken:.2f} seconds")
    print(f"Total Trials: {n_trials}")
    
    results = save_results(
        method_name=f'Optuna ({model_type})',
        best_params=best_params,
        metrics=metrics,
        time_taken=time_taken,
        n_iterations=n_trials
    )
    
    return best_model, results

if __name__ == "__main__":
    from utils import load_wine_data, prepare_data
    
    df = load_wine_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    run_optuna_optimization(X_train, y_train, X_test, y_test, model_type='random_forest')
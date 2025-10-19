import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils import load_wine_data, prepare_data
from optimization.grid_search import run_grid_search
from optimization.random_search import run_random_search
from optimization.bayesian_opt import run_bayesian_optimization
from optimization.optuna_opt import run_optuna_optimization
from compare_methods import load_all_results, create_comparison_plots, create_summary_table

def run_all_optimizations(model_type='random_forest', n_iter=30):
    
    print("WINE QUALITY PREDICTION - HYPERPARAMETER OPTIMIZATION COMPARISON")
    print("="*70 + "\n")
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_wine_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Model type: {model_type.upper()}\n")
    
    all_results = []
    
    # Grid Search
    try:
        print("\n[1/4] Starting Grid Search...")
        _, result = run_grid_search(X_train, y_train, X_test, y_test, model_type)
        all_results.append(result)
    except Exception as e:
        print(f"Error in Grid Search: {e}")
    
    # Random Search
    try:
        print("\n[2/4] Starting Random Search...")
        _, result = run_random_search(X_train, y_train, X_test, y_test, model_type, n_iter)
        all_results.append(result)
    except Exception as e:
        print(f"Error in Random Search: {e}")
    
    # Bayesian Optimization
    try:
        print("\n[3/4] Starting Bayesian Optimization...")
        _, result = run_bayesian_optimization(X_train, y_train, X_test, y_test, model_type, n_iter)
        all_results.append(result)
    except Exception as e:
        print(f"Error in Bayesian Optimization: {e}")
    
    # Optuna
    try:
        print("\n[4/4] Starting Optuna Optimization...")
        _, result = run_optuna_optimization(X_train, y_train, X_test, y_test, model_type, n_iter)
        all_results.append(result)
    except Exception as e:
        print(f"Error in Optuna: {e}")
    
    # Generate comparison plots and summary
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS AND SUMMARY")
    print("="*70 + "\n")
    
    results = load_all_results()
    if results:
        create_comparison_plots(results)
        create_summary_table(results)
    
    print("\n" + "="*70)
    print("ALL OPTIMIZATIONS COMPLETED!")
    print("="*70 + "\n")
    print("Check the 'results' directory for detailed outputs.")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization comparison')
    parser.add_argument('--model', type=str, default='random_forest', 
                        choices=['random_forest', 'xgboost'],
                        help='Model type to use (default: random_forest)')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Number of iterations for search methods (default: 30)')
    
    args = parser.parse_args()
    
    run_all_optimizations(model_type=args.model, n_iter=args.iterations)
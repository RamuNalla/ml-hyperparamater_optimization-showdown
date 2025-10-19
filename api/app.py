from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import sys
import numpy as np
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_wine_data, prepare_data
from optimization.grid_search import run_grid_search
from optimization.random_search import run_random_search
from optimization.bayesian_opt import run_bayesian_optimization
from optimization.optuna_opt import run_optuna_optimization
from compare_methods import load_all_results

# Global variables to store data and model
X_train, X_test, y_train, y_test, scaler, trained_model = None, None, None, None, None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup and clean up on shutdown."""
    global X_train, X_test, y_train, y_test, scaler
    print("Loading data on application startup...")
    try:
        df = load_wine_data()
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
    yield

app = FastAPI(title="Wine Quality ML Optimizer API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizationRequest(BaseModel):
    method: str
    model_type: str = "random_forest"
    n_iterations: int = 30

class PredictionRequest(BaseModel):
    features: List[float]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Wine Quality ML Optimizer API",
        "version": "1.0.0",
        "endpoints": {
            "/optimize": "Run hyperparameter optimization",
            "/results": "Get all optimization results",
            "/results/{method}": "Get results for specific method",
            "/dataset-info": "Get dataset information"
        }
    }

@app.get("/dataset-info")
async def get_dataset_info():
    """Get information about the dataset"""
    if X_train is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return {
        "training_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "n_features": X_train.shape[1],
        "class_distribution": {
            "train": {
                "class_0": int((y_train == 0).sum()),
                "class_1": int((y_train == 1).sum())
            },
            "test": {
                "class_0": int((y_test == 0).sum()),
                "class_1": int((y_test == 1).sum())
            }
        }
    }

@app.post("/optimize")
async def optimize(request: OptimizationRequest):
    """Run hyperparameter optimization"""
    if X_train is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        method = request.method.lower()
        model_type = request.model_type.lower()
        n_iter = request.n_iterations
        
        if method == "grid_search":
            _, result = run_grid_search(X_train, y_train, X_test, y_test, model_type)
        elif method == "random_search":
            _, result = run_random_search(X_train, y_train, X_test, y_test, model_type, n_iter)
        elif method == "bayesian":
            _, result = run_bayesian_optimization(X_train, y_train, X_test, y_test, model_type, n_iter)
        elif method == "optuna":
            _, result = run_optuna_optimization(X_train, y_train, X_test, y_test, model_type, n_iter)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
        
        return {
            "status": "success",
            "method": method,
            "model_type": model_type,
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/results")
async def get_all_results():
    """Get all optimization results"""
    try:
        results = load_all_results()
        if not results:
            return {
                "status": "no_results",
                "message": "No optimization results found. Run optimizations first.",
                "results": []
            }
        
        return {
            "status": "success",
            "count": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/results/{method}")
async def get_result_by_method(method: str):
    """Get results for a specific optimization method"""
    try:
        results_dir = 'results'
        filename = f"{method.lower().replace(' ', '_')}_results.json"
        filepath = os.path.join(results_dir, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"Results not found for method: {method}")
        
        with open(filepath, 'r') as f:
            result = json.load(f)
        
        return {
            "status": "success",
            "result": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/methods")
async def get_available_methods():
    """Get list of available optimization methods"""
    return {
        "methods": [
            {
                "name": "grid_search",
                "description": "Exhaustive search over parameter grid",
                "pros": ["Guaranteed to find best combination in grid", "Easy to understand"],
                "cons": ["Computationally expensive", "Doesn't scale well"]
            },
            {
                "name": "random_search",
                "description": "Random sampling from parameter distributions",
                "pros": ["More efficient than grid search", "Works well with large search spaces"],
                "cons": ["May miss optimal configuration", "No learning from previous trials"]
            },
            {
                "name": "bayesian",
                "description": "Bayesian optimization using Gaussian Processes",
                "pros": ["Efficient exploration", "Uses previous results to guide search"],
                "cons": ["More complex", "Can be slower per iteration"]
            },
            {
                "name": "optuna",
                "description": "Tree-structured Parzen Estimator (TPE)",
                "pros": ["Modern and efficient", "Supports pruning", "Easy to use"],
                "cons": ["Requires separate library", "Black box approach"]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
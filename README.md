# ML Hyperparameter Optimization Showdown

A comprehensive ML project comparing multiple hyperparameter optimization methods on wine quality prediction. This project includes a FastAPI backend, Streamlit web interface, and detailed comparisons of Grid Search, Random Search, Bayesian Optimization, and Optuna.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This project demonstrates the practical application of various hyperparameter optimization techniques on a real-world classification problem. Using the Wine Quality dataset from UCI ML Repository, we compare different optimization methods in terms of:

- **Accuracy**: Model performance on test data
- **Speed**: Time taken to complete optimization
- **Efficiency**: Performance per unit time
- **Convergence**: How quickly methods find optimal parameters

## Features

- **4 Optimization Methods**: Grid Search, Random Search, Bayesian Optimization, and Optuna
- **2 ML Models**: Random Forest and XGBoost classifiers
- **REST API**: FastAPI backend for programmatic access
- **Web Interface**: Interactive Streamlit dashboard
- **Visualization**: Comprehensive comparison plots and metrics
- **Reproducibility**: Fixed random seeds for consistent results
- **Extensibility**: Easy to add new models and optimization methods

## Project Structure

```
ml-hyperparamater_optimization-showdown/
│
├── src/                          # Source code
│   ├── utils.py                  # Utility functions
│   ├── main.py                   # Main execution script
│   ├── compare_methods.py        # Comparison and visualization
│   ├── results/                      # Results directory
│       ├── *_results.json           # Individual method results
│       ├── comparison_plot.png      # Comparison visualization
│       └── summary_table.csv        # Summary statistics
│   └── optimization/             # Optimization methods
│       ├── __init__.py
│       ├── grid_search.py
│       ├── random_search.py
│       ├── bayesian_opt.py
│       └── optuna_opt.py
│
├── api/                          # FastAPI application
│   └── app.py                    # API endpoints
│   └── streamlit_app.py          # Web interface
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                   # Git ignore file
```

##  Installation

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/RamuNalla/ml-hyperparamater_optimization-showdown.git
cd ml-hyperparamater_optimization-showdown

```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Run All Optimizations (Command Line)

Run all optimization methods with default settings:

```bash
cd src
python main.py
```

Run with custom parameters:

```bash
python main.py --model xgboost --iterations 50
```

### Method 2: Run Individual Methods

```bash
cd src/optimization
python grid_search.py
python random_search.py
python bayesian_opt.py
python optuna_opt.py
```

### Method 3: Use the API

1. **Start the FastAPI server**

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

2. **Access the API documentation**

Open your browser and navigate to:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`


### Method 3: Use the Streamlit Web Interface

1. **Start the FastAPI server** (in one terminal)

```bash
uvicorn api.app:app --reload
```

2. **Start the Streamlit app** (in another terminal)

```bash
cd app
streamlit run streamlit_app.py
```

3. **Open your browser**

Navigate to `http://localhost:8501`

## Optimization Methods

### 1. Grid Search

**Description**: Exhaustively searches through a manually specified subset of hyperparameter space.

**Pros**:
- Guaranteed to find the best combination within the grid
- Simple and easy to understand
- Deterministic results

**Cons**:
- Computationally expensive
- Doesn't scale well with number of parameters
- Can be inefficient with continuous parameters

**Use Case**: Small parameter spaces where you want to be thorough

### 2. Random Search

**Description**: Randomly samples from the parameter distribution for a fixed number of iterations.

**Pros**:
- More efficient than grid search
- Works well with continuous parameters
- Can be easily parallelized

**Cons**:
- May miss the optimal configuration
- No learning from previous trials
- Results vary between runs

**Use Case**: Large parameter spaces with limited computational budget

### 3. Bayesian Optimization

**Description**: Uses probabilistic models (Gaussian Processes) to model the objective function and select promising hyperparameters.

**Pros**:
- Efficient exploration of parameter space
- Uses past evaluation results intelligently
- Good for expensive objective functions

**Cons**:
- More complex to implement and understand
- Can be slow for high-dimensional spaces
- Requires more setup

**Use Case**: When function evaluations are expensive and you want smart exploration

### 4. Optuna (TPE)

**Description**: Uses Tree-structured Parzen Estimator (TPE) algorithm with built-in pruning capabilities.

**Pros**:
- Modern and user-friendly API
- Supports early stopping (pruning)
- Excellent performance in practice
- Great visualization tools

**Cons**:
- Another dependency to manage
- Black box approach
- Less theoretical guarantees than Bayesian methods

**Use Case**: Default choice for modern ML projects, especially with deep learning

## Results

After running the optimizations, you'll find:

1. **JSON files** in `results/` directory containing detailed results for each method
2. **Comparison plot** (`comparison_plot.png`) showing visual comparison
3. **Summary table** (`summary_table.csv`) with key metrics

### Sample Output

```
Method                          | Accuracy | F1 Score | Time (s) | Iterations
Grid Search (random_forest)     | 0.8187   | 0.8188   | 199.24   | 108
Random Search (random_forest)   | 0.7937   | 0.7939   | 89.79    | 30
Bayesian (random_forest)        | 0.8156   | 0.8120   | 89.45    | 30
Optuna (random_forest)          | 0.8063   | 0.8063   | 58.54    | 30
```

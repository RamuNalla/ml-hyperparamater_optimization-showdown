import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page config
st.set_page_config(
    page_title="Wine Quality ML Optimizer",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #8B0000;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🍷 Wine Quality ML Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hyperparameter Optimization Comparison Platform</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    optimization_method = st.selectbox(
        "Select Optimization Method",
        ["Grid Search", "Random Search", "Bayesian Optimization", "Optuna"],
        help="Choose the hyperparameter optimization method to run"
    )
    
    model_type = st.selectbox(
        "Select Model Type",
        ["Random Forest", "XGBoost"],
        help="Choose the machine learning model to optimize"
    )
    
    n_iterations = st.slider(
        "Number of Iterations",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Number of iterations for search methods (not applicable to Grid Search)"
    )
    
    st.markdown("---")
    
    run_optimization = st.button("🚀 Run Optimization", use_container_width=True, type="primary")
    view_results = st.button("📊 View All Results", use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Tip**: Start with 30 iterations for a good balance between speed and performance.")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔬 Run Optimization", "📊 Results", "ℹ️ About"])


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

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Project Overview")
        st.write("""
        This application demonstrates and compares various hyperparameter optimization techniques 
        for machine learning models. Using the Wine Quality dataset, we optimize classification 
        models to predict wine quality ratings.
        
        **Features:**
        - 4 optimization methods
        - 2 model types (Random Forest & XGBoost)
        - Real-time performance comparison
        - Interactive visualizations
        """)
        
        st.subheader("🎯 Dataset Information")
        try:
            response = requests.get(f"{API_URL}/dataset-info")
            if response.status_code == 200:
                data_info = response.json()
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Training Samples", data_info['training_samples'])
                with col_b:
                    st.metric("Test Samples", data_info['test_samples'])
                with col_c:
                    st.metric("Features", data_info['n_features'])
            else:
                st.warning("API not available. Make sure the FastAPI server is running.")
        except:
            st.warning("⚠️ Cannot connect to API. Please start the FastAPI server first.")
    
    with col2:
        st.subheader("🔍 Optimization Methods")
        
        methods_info = [
            ("Grid Search", "Exhaustive search over parameter grid", "🟢 Thorough", "🔴 Slow"),
            ("Random Search", "Random sampling from distributions", "🟢 Fast", "🟡 Random"),
            ("Bayesian Optimization", "Uses Gaussian Processes", "🟢 Smart", "🟡 Complex"),
            ("Optuna", "TPE algorithm with pruning", "🟢 Modern", "🟢 Efficient")
        ]
        
        for name, desc, pro, con in methods_info:
            with st.expander(f"**{name}**"):
                st.write(desc)
                st.write(f"**Pros:** {pro}")
                st.write(f"**Cons:** {con}")

with tab2:
    st.subheader("🔬 Run Hyperparameter Optimization")
    
    if run_optimization:
        with st.spinner(f"Running {optimization_method} optimization... This may take a few minutes."):
            try:
                method_mapping = {
                    "Grid Search": "grid_search",
                    "Random Search": "random_search",
                    "Bayesian Optimization": "bayesian",
                    "Optuna": "optuna"
                }
                
                model_mapping = {
                    "Random Forest": "random_forest",
                    "XGBoost": "xgboost"
                }
                
                payload = {
                    "method": method_mapping[optimization_method],
                    "model_type": model_mapping[model_type],
                    "n_iterations": n_iterations
                }
                
                response = requests.post(f"{API_URL}/optimize", json=payload)
                
                if response.status_code == 200:
                    result = response.json()['result']
                    
                    st.success(f"✅ {optimization_method} completed successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{result['metrics']['accuracy']:.4f}")
                    with col2:
                        st.metric("F1 Score", f"{result['metrics']['f1_score']:.4f}")
                    with col3:
                        st.metric("Time Taken", f"{result['time_taken']:.2f}s")
                    with col4:
                        st.metric("Iterations", result['n_iterations'])
                    
                    st.subheader("🎯 Best Parameters")
                    params_df = pd.DataFrame([result['best_params']]).T
                    params_df.columns = ['Value']
                    st.dataframe(params_df, use_container_width=True)
                    
                else:
                    st.error(f"Error: {response.json()['detail']}")
            
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
    else:
        st.info("👈 Configure your optimization settings in the sidebar and click 'Run Optimization'")


with tab3:
    st.subheader("📊 Optimization Results Comparison")
    
    try:
        response = requests.get(f"{API_URL}/results")
        
        if response.status_code == 200:
            data = response.json()
            
            if data['status'] == 'no_results':
                st.warning("No results available yet. Run some optimizations first!")
            else:
                results = data['results']
                
                # Create summary dataframe
                summary_data = []
                for r in results:
                    summary_data.append({
                        'Method': r['method'],
                        'Accuracy': r['metrics']['accuracy'],
                        'F1 Score': r['metrics']['f1_score'],
                        'Time (s)': r['time_taken'],
                        'Iterations': r['n_iterations']
                    })
                
                df = pd.DataFrame(summary_data)
                
                # Display summary table
                st.subheader("📈 Summary Table")
                st.dataframe(df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score']), 
                           use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Accuracy comparison
                    fig_acc = px.bar(df, x='Method', y='Accuracy', 
                                    title='Accuracy Comparison',
                                    color='Accuracy',
                                    color_continuous_scale='Viridis')
                    fig_acc.update_layout(showlegend=False)
                    st.plotly_chart(fig_acc, use_container_width=True)
                    
                    # Time comparison
                    fig_time = px.bar(df, x='Method', y='Time (s)', 
                                     title='Execution Time Comparison',
                                     color='Time (s)',
                                     color_continuous_scale='Reds')
                    fig_time.update_layout(showlegend=False)
                    st.plotly_chart(fig_time, use_container_width=True)
                
                with col2:
                    # F1 Score comparison
                    fig_f1 = px.bar(df, x='Method', y='F1 Score', 
                                   title='F1 Score Comparison',
                                   color='F1 Score',
                                   color_continuous_scale='Blues')
                    fig_f1.update_layout(showlegend=False)
                    st.plotly_chart(fig_f1, use_container_width=True)
                    
                    # Efficiency (Accuracy per second)
                    df['Efficiency'] = df['Accuracy'] / (df['Time (s)'] + 1)
                    fig_eff = px.bar(df, x='Method', y='Efficiency', 
                                    title='Efficiency (Accuracy / Time)',
                                    color='Efficiency',
                                    color_continuous_scale='Greens')
                    fig_eff.update_layout(showlegend=False)
                    st.plotly_chart(fig_eff, use_container_width=True)
                
                # Radar chart
                st.subheader("🎯 Multi-Metric Comparison")
                
                # Normalize metrics for radar chart
                df_norm = df.copy()
                df_norm['Accuracy'] = (df_norm['Accuracy'] - df_norm['Accuracy'].min()) / (df_norm['Accuracy'].max() - df_norm['Accuracy'].min())
                df_norm['F1 Score'] = (df_norm['F1 Score'] - df_norm['F1 Score'].min()) / (df_norm['F1 Score'].max() - df_norm['F1 Score'].min())
                df_norm['Speed'] = 1 - ((df_norm['Time (s)'] - df_norm['Time (s)'].min()) / (df_norm['Time (s)'].max() - df_norm['Time (s)'].min()))
                
                fig_radar = go.Figure()
                
                for idx, row in df_norm.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row['Accuracy'], row['F1 Score'], row['Speed']],
                        theta=['Accuracy', 'F1 Score', 'Speed'],
                        fill='toself',
                        name=row['Method']
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Normalized Performance Metrics"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
        
        else:
            st.error("Error fetching results from API")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

with tab4:
    st.subheader("ℹ️ About This Project")
    
    st.markdown("""
    ### 🍷 Wine Quality ML Optimizer
    
    This project demonstrates various hyperparameter optimization techniques for machine learning models.
    
    **Technologies Used:**
    - **FastAPI**: REST API backend
    - **Streamlit**: Interactive web interface
    - **Scikit-learn**: Machine learning models
    - **XGBoost**: Gradient boosting framework
    - **Optuna**: Modern hyperparameter optimization
    - **Plotly**: Interactive visualizations
    
    **Optimization Methods:**
    1. **Grid Search**: Systematic exploration of parameter space
    2. **Random Search**: Random sampling of parameters
    3. **Bayesian Optimization**: Probabilistic model-based optimization
    4. **Optuna**: Tree-structured Parzen Estimator (TPE)
    
    **Dataset:**
    - Wine Quality Dataset from UCI ML Repository
    - Binary classification (good vs. bad wine)
    - 11 physicochemical features
      
    **Author**: Ramu Nalla
    """)
    
    st.markdown("---")
    st.info("🚀 Start the FastAPI server with: `uvicorn api.app:app --reload`")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>Made with ❤️ using Streamlit and FastAPI</div>",
    unsafe_allow_html=True
)


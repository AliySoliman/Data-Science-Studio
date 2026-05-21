import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from constants import DataManager
import transform
from models import model as model
data_manager = DataManager()

def run():
        
        # Custom CSS for the entire application
        st.markdown("""
        <style>
            /* Model card styling */
            .model-card {
                border-radius: 8px;
                padding: 12px 16px;
                margin: 8px 0;
                background-color: #123B5F;
                border-left: 4px solid #4e79a7;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .model-card:hover {
                background-color: #3941AA;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .initials-badge {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                color: white;
                margin-right: 12px;
                flex-shrink: 0;
                background-color: #4e79a7;
            }
            
            /* Code editor-like container */
            .code-container {
                background-color: #0e2440;
                border-radius: 8px;
                padding: 16px;
                margin: 16px 0;
                font-family: 'Courier New', monospace;
                border: 1px solid #e0e0e0;
            }
            .code-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
                color: #FFFFFF;
                font-size: 0.9em;
            }
            
            /* Report styling */
            .report-container {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 16px;
                margin-top: 20px;
            }
            .metric-card {
                background-color: #f8f9fa;
                border-radius: 6px;
                padding: 12px;
                margin: 8px 0;
            }
        </style>
        """, unsafe_allow_html=True)

        # Sample data for the data transformation dialog
        


        def get_initials(name):
            """Get initials from model name"""
            words = name.split()
            if len(words) >= 2:
                return words[0][0] + words[-1][0]
            return name[:2]

        def get_color(name):
            """Get consistent color based on model name"""
            colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
            return colors[hash(name) % len(colors)]


        def create_model_card(col, model_name, subcategory):
            """Create a card for a single model"""
            initials = get_initials(model_name)
            color = get_color(model_name)
            button_text = "Edit" if model_name in performed_models else "Create"
            
            with col:
                # Card container
                                            # <div class="card-type">{subcategory}</div>

                card = st.container()
                card.markdown(
                    f"""
                    <div class="model-card">
                        <div class="initials-badge" style="background-color: {color}">
                            {initials}
                        </div>
                        <div class="card-content">
                            <div class="card-title">{model_name}</div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Button column (aligned to right)
                if card.button(button_text, key=f"btn_{model_name}"):
                    st.session_state.current_page = model_name
                    st.rerun()

        def main_page():
            """Main page with expander categories and card-style models"""
            st.markdown(data_manager.label_style, unsafe_allow_html=True)
            st.markdown("""
            <div class="custom-container">
                <h2 class="custom-header">🚀 Machine Learning Studio</h2>
            </div>
            """, unsafe_allow_html=True)
            # st.title("Machine Learning Studio")
            
            # Filter checkbox
            show_only_performed = st.checkbox("Show only performed models", value=False)
            
            # Display each category as an expander
            for category, subcategories in model_categories.items():
                with st.expander(category, expanded=True):
                    for subcategory, models in subcategories.items():
                        st.markdown(f'<div style="font-weight:bold; margin:12px 0 8px 0; color:#FFFFFF">{subcategory}</div>', unsafe_allow_html=True)
                        
                        # Create columns for the cards (3 per row)
                        cols = st.columns(3)
                        col_idx = 0
                        
                        for m_name in models:
                            if show_only_performed and m_name not in performed_models:
                                continue
                            
                            create_model_card(cols[col_idx], m_name, subcategory)
                            col_idx = (col_idx + 1) % 3
                        
                        # Add divider between subcategories if there were any models shown
                        if any(not show_only_performed or m_name in performed_models for m_name in models):
                            st.markdown('<hr style="margin:8px 0">', unsafe_allow_html=True)

        # Model categories and performed models

        model_categories ={
            "Regression": {
                "Standard Regression": ["Linear Regression", "KNN_Regressor", "Support Vector Regression"],
                "Ensemble Methods": ["Random Forest Regression", "Gradient Boosting Regressor", "XGBoost Regressor"]
            },
            "Unsupervised": {
                "Clustering": ["Hierarchical Clustering", "KMeans Clustering", "Unsupervised KNN Clustering", "DBSCAN"]
            },
            "Classification": {
                "Linear & Statistical": ["Logistic Regression", "Naive Bayes Classifier", "Support Vector Machine Classifier"],
                "Tree-Based": ["Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier", "XGBoost Classifier"],
                "Neighbors": ["KNN Classifier"]
            }
        }
        # st.write(st.session_state.pipeline['ML'])
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline:
            st.session_state.pipeline['ML'] = []
            
        performed_models = [model['model type'] for model in  st.session_state.pipeline['ML']]

        # Initialize session state for page navigation
        if "current_page" not in st.session_state:
            st.session_state.current_page = "main"

        # Page routing
        if st.session_state.current_page == "main":
            main_page()
        else:
            data_uploaded = st.session_state.load == False
            edit =  st.session_state.current_page in performed_models 
            model_data = {}
            if edit :
                for saved_model in st.session_state.pipeline['ML']:
                    #  st.write(saved_model)
                     if saved_model['model type'] == st.session_state.current_page:
                        model_data =  saved_model
                # model_data = [ model for model in st.session_state.pipeline['ML'] if model['model type'] == st.session_state.current_page]
                # st.write(model_data)
            
     
            model.show_model_page(
                edit,
                model_data
                ,st.session_state.current_page,
                data_uploaded,
                True
                )
def delete_model(model_name):
            for i,saved_model in enumerate(st.session_state.pipeline['ML']) :
                if saved_model['model name'] == model_name :
                    st.session_state.pipeline['ML'].pop(i)

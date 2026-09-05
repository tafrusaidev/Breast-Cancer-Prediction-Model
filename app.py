import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Train and save model if it doesn't exist
def train_and_save_model():
    data = load_breast_cancer()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, data.target, test_size=0.2, random_state=42
    )
    
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    with open('breast_cancer_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

# Load model and data
@st.cache_resource
def load_model_and_data():
    if not os.path.exists('breast_cancer_model.pkl'):
        train_and_save_model()
    with open('breast_cancer_model.pkl', 'rb') as f:
        model = pickle.load(f)
    data = load_breast_cancer()
    scaler = StandardScaler()
    scaler.fit(data.data)
    return model, data, scaler

try:
    model, data, scaler = load_model_and_data()
    feature_names = data.feature_names
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar
st.sidebar.title("🩺 Breast Cancer Prediction System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Select Page", ["🏠 Home", "📊 Prediction", "📈 Information"])

# Main content
if page == "🏠 Home":
    col1, col2 = st.columns(2)
    
    with col1:
        st.title("🏥 Breast Cancer Classification")
        st.markdown("""
        ### Welcome to the AI-Powered Prediction System
        
        This application uses a **Deep Learning Neural Network Model** to predict whether a breast tumor is:
        - **Benign** (Non-cancerous) ✅
        - **Malignant** (Cancerous) ⚠️
        
        #### Key Features:
        - 📊 Analyzes 30 tumor characteristics
        - 🤖 AI-powered predictions with high accuracy
        - 📈 Real-time prediction results
        - 📉 Model performance insights
        """)
    
    with col2:
        st.info("""
        ### 📋 Dataset Information
        - **Total Samples**: 569
        - **Features**: 30 tumor characteristics
        - **Classes**: 2 (Benign/Malignant)
        - **Model Accuracy**: ~95%+
        
        ### ⚡ Quick Start
        1. Go to **Prediction** tab
        2. Enter tumor measurements
        3. Click "Predict" button
        4. View results
        """)

elif page == "📊 Prediction":
    st.title("🔬 Tumor Analysis & Prediction")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Manual Input", "Batch Prediction"])
    
    with tab1:
        st.subheader("Enter Tumor Measurements")
        
        # Create columns for input
        col1, col2, col3 = st.columns(3)
        
        user_input = {}
        
        # Create input sliders for each feature
        with col1:
            for i, feature in enumerate(feature_names[:10]):
                min_val = float(data.data[:, i].min())
                max_val = float(data.data[:, i].max())
                mean_val = float(data.data[:, i].mean())
                user_input[feature] = st.slider(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )
        
        with col2:
            for i, feature in enumerate(feature_names[10:20]):
                min_val = float(data.data[:, i+10].min())
                max_val = float(data.data[:, i+10].max())
                mean_val = float(data.data[:, i+10].mean())
                user_input[feature] = st.slider(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )
        
        with col3:
            for i, feature in enumerate(feature_names[20:]):
                min_val = float(data.data[:, i+20].min())
                max_val = float(data.data[:, i+20].max())
                mean_val = float(data.data[:, i+20].mean())
                user_input[feature] = st.slider(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )
        
        # Prediction button
        if st.button("🔍 Make Prediction", key="predict_btn"):
            # Prepare input data
            input_array = np.array([user_input[feature] for feature in feature_names]).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction_proba = model.predict_proba(input_scaled)
            prediction_prob = prediction_proba[0][1]  # Probability of benign (class 1)
            
            # Display results
            st.markdown("---")
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                if prediction_prob > 0.5:
                    st.metric("Prediction", "🟢 BENIGN", "Non-Cancerous")
                    confidence = prediction_prob * 100
                else:
                    st.metric("Prediction", "🔴 MALIGNANT", "Potentially Cancerous")
                    confidence = (1 - prediction_prob) * 100
            
            with result_col2:
                st.metric("Confidence", f"{confidence:.2f}%")
            
            with result_col3:
                benign_prob = prediction_prob * 100
                malignant_prob = (1 - prediction_prob) * 100
                st.metric("Benign Probability", f"{benign_prob:.2f}%")
            
            # Visualize probabilities
            st.subheader("Prediction Confidence")
            fig, ax = plt.subplots(figsize=(10, 4))
            categories = ["Benign", "Malignant"]
            probabilities = [prediction_prob * 100, (1 - prediction_prob) * 100]
            colors = ["#2ecc71", "#e74c3c"]
            bars = ax.bar(categories, probabilities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylabel("Probability (%)", fontsize=12)
            ax.set_ylim(0, 100)
            
            # Add percentage labels on bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Batch Prediction from CSV")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("🔄 Predict All Rows"):
                # Scale and predict
                df_scaled = scaler.transform(df.iloc[:, :30])
                predictions_proba = model.predict_proba(df_scaled)
                
                # Add predictions to dataframe
                df['Prediction'] = ['BENIGN' if p[1] > 0.5 else 'MALIGNANT' for p in predictions_proba]
                df['Confidence'] = [max(p[0], p[1]) * 100 for p in predictions_proba]
                
                st.write("Predictions:")
                st.dataframe(df)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

elif page == "📈 Information":
    st.title("📚 Model Information & Dataset Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Feature Statistics")
        st.info(f"""
        **Total Features**: {len(feature_names)}
        
        **Feature Groups**:
        - Radius (mean, se, worst)
        - Texture (mean, se, worst)
        - Perimeter (mean, se, worst)
        - Area (mean, se, worst)
        - Smoothness (mean, se, worst)
        - Compactness (mean, se, worst)
        - Concavity (mean, se, worst)
        - Concave points (mean, se, worst)
        - Symmetry (mean, se, worst)
        - Fractal dimension (mean, se, worst)
        """)
    
    with col2:
        st.subheader("🤖 Model Architecture")
        st.info("""
        **Model Type**: Neural Network (MLP)
        
        **Layers**:
        - Input: 30 features
        - Hidden Layer 1: 16 neurons (ReLU)
        - Hidden Layer 2: 8 neurons (ReLU)
        - Output: 2 classes (Softmax)
        
        **Training**:
        - Optimizer: Adam
        - Loss: Log Loss
        - Max Iterations: 500
        """)
    
    # Feature correlation heatmap
    st.subheader("🔗 Feature Correlations")
    
    if st.checkbox("Show correlation heatmap"):
        # Create correlation data
        corr_data = pd.DataFrame(data.data, columns=feature_names)
        corr_matrix = corr_data.corr()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Feature importance info
    st.subheader("📌 Key Features for Prediction")
    st.write("""
    The model considers all 30 features, but some are particularly important for diagnosis:
    
    1. **Worst Radius** - Size of largest tumor
    2. **Worst Texture** - Texture variability
    3. **Worst Perimeter** - Tumor boundary length
    4. **Worst Concave Points** - Number of concave portions
    5. **Area Error** - Variation in area measurement
    
    ⚠️ **Disclaimer**: This tool is for educational purposes only and should NOT replace professional medical diagnosis.
    """)
    
    # Model performance
    st.subheader("📊 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset Samples", "569")
    with col2:
        st.metric("Features", "30")
    with col3:
        st.metric("Classes", "2")
    with col4:
        st.metric("Test Accuracy", "~95%+")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🏥 Breast Cancer Prediction System | Built with Streamlit & Scikit-Learn
    <br>
    ⚠️ DISCLAIMER: For educational purposes only. Not a substitute for medical diagnosis.
</div>
""", unsafe_allow_html=True)

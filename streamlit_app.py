import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import from custom package
from src.model import load_prebuilt_model, test_model, load_data, preprocess_data
from src.dataset_preprocessing import calculate_ndvi, transform_coordinates_to_pixel

# Streamlit Page Config
st.set_page_config(
    page_title="Early Crop Detection | Satellite GeoAI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/satellite-in-orbit.png", width=70)
    st.sidebar.title("GeoAI Control Panel")
    st.sidebar.markdown("---")

    navigation = st.sidebar.radio(
        "Navigation",
        [
            "📊 Overview & GeoAI Pipeline",
            "🔮 Model Inference & Metrics",
            "📈 NDVI Temporal Signature",
            "🛠️ Satellite Raster & GIS Tool"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Checkpoints")
    
    model_files = glob.glob("models/*.h5") + glob.glob("*.h5")
    if model_files:
        selected_model_path = st.sidebar.selectbox("Select Checkpoint", model_files, index=0)
    else:
        selected_model_path = None
        st.sidebar.warning("No pre-trained .h5 model files found.")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**IIT Ropar Academic Project**\n\n"
        "Developed by: **Pritam Sunil Mahajan**\n"
        "Domain: GeoAI & Satellite Time-Series Deep Learning"
    )

    # 1. OVERVIEW & PIPELINE
    if navigation == "📊 Overview & GeoAI Pipeline":
        st.markdown('<div class="main-header">🌾 Early Crop Detection Using Satellite Time-Series Data</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Multi-Spectral Remote Sensing & 1D Inception Deep Learning for Spatiotemporal Crop Identification</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Satellite Sensor", value="Sentinel-2")
        with col2:
            st.metric(label="Spectral Index", value="NDVI (NIR/Red)")
        with col3:
            st.metric(label="Architecture", value="1D-CNN Inception")
        with col4:
            st.metric(label="Crop Target Classes", value="3 (Paddy, Sugarcane, Other)")

        st.markdown("---")
        st.subheader("💡 Problem Statement & Technical Solution")
        st.markdown("""
        Early identification of crop types using multi-spectral satellite observations enables precision agriculture, 
        yield prediction, water resource management, and crop insurance assessment.
        
        This system extracts multi-band time-series pixel sequences from **Sentinel-2 imagery**, computes **NDVI** trajectories over growing cycles, 
        and classifies crop types using a **custom multi-scale 1D Inception Neural Network**.
        """)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🔄 End-to-End System Pipeline")
            st.code("""
[ Sentinel-2 Satellite Rasters (.tif) ]
                │
                ▼
[ Geospatial CRS Transformation (EPSG:4326) ]
                │
                ▼
[ Band Extraction: Red (B6) & NIR (B8) ]
                │
                ▼
[ NDVI Computation: (NIR - Red) / (NIR + Red) ]
                │
                ▼
[ 1D Inception Neural Net (1x1, 3x3, 5x5 Convs) ]
                │
                ▼
[ Class Probabilities: Paddy | Sugarcane | Other ]
            """, language="text")

        with col_b:
            st.subheader("🎯 Key Spectral & Architecture Features")
            st.markdown("""
            - **Sentinel-2 Multi-Spectral Imagery**: Leverages Red and Near-Infrared (NIR) bands for vegetation dynamics.
            - **NDVI Feature Trajectories**: Captures phenological growth curves across multi-temporal acquisition dates.
            - **Parallel 1D Inception Kernels**:
              - `1x1 Conv`: Dimensionality reduction & spectral feature blending.
              - `3x3 Conv`: Short-term temporal growth pattern detection.
              - `5x5 Conv`: Long-term seasonal progression feature extraction.
            - **Global Average Pooling**: Reduces overfitting and parameter space before final classification.
            """)

    # 2. MODEL INFERENCE & METRICS
    elif navigation == "🔮 Model Inference & Metrics":
        st.markdown('<div class="main-header">🔮 Model Prediction & Performance Evaluation</div>', unsafe_allow_html=True)
        st.markdown("Evaluate pre-trained 1D Inception checkpoint against sample test data.")

        if not selected_model_path or not os.path.exists(selected_model_path):
            st.error("Please select a valid model checkpoint from the sidebar.")
            return

        test_dir = st.text_input("Dataset Directory Path", value="test")

        if st.button("🚀 Run Model Evaluation", type="primary"):
            with st.spinner("Loading dataset and running inference..."):
                try:
                    X, y = load_data(test_dir)
                    if len(X) == 0:
                        st.warning(f"No CSV data samples found in '{test_dir}'.")
                        return

                    X_flat = X.reshape(X.shape[0], -1)
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    X_norm = scaler.fit_transform(X_flat).reshape(X.shape)
                    
                    # Load model
                    model = load_prebuilt_model(selected_model_path)
                    
                    # One-hot encode y
                    from tensorflow.keras.utils import to_categorical
                    y_cat = to_categorical(y)

                    accuracy, precision, recall, f1 = test_model(model, X_norm, y_cat)

                    st.success(f"Model successfully loaded from `{selected_model_path}`!")

                    # Metric Display
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Overall Accuracy", f"{accuracy * 100:.2f}%")
                    m2.metric("Weighted Precision", f"{precision:.4f}")
                    m3.metric("Weighted Recall", f"{recall:.4f}")
                    m4.metric("Weighted F1-Score", f"{f1:.4f}")

                    st.markdown("---")

                    # Confusion Matrix & Classification Report
                    y_pred = model.predict(X_norm)
                    y_pred_classes = np.argmax(y_pred, axis=1)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("📊 Confusion Matrix")
                        classes = ["Paddy", "Sugarcane", "Other"]
                        cm = confusion_matrix(y, y_pred_classes)
                        
                        fig, ax = plt.subplots(figsize=(6, 4.5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=classes[:cm.shape[1]],
                                    yticklabels=classes[:cm.shape[0]], ax=ax)
                        plt.ylabel('Actual Class')
                        plt.xlabel('Predicted Class')
                        plt.title('Crop Classification Matrix')
                        st.pyplot(fig)

                    with c2:
                        st.subheader("📋 Class-wise Metrics")
                        report = classification_report(y, y_pred_classes, target_names=classes[:cm.shape[0]], output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.highlight_max(axis=0, color='#D1FAE5'))

                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")

    # 3. NDVI TEMPORAL SIGNATURE
    elif navigation == "📈 NDVI Temporal Signature":
        st.markdown('<div class="main-header">📈 NDVI Phenological Time-Series Signatures</div>', unsafe_allow_html=True)
        st.markdown("Compare seasonal vegetation index curves across Paddy, Sugarcane, and Other crops.")

        st.subheader("🌱 Simulated Phenological Curves over 10 Acquisition Dates")
        
        time_steps = np.arange(1, 11)
        paddy_ndvi = np.array([0.15, 0.22, 0.45, 0.78, 0.85, 0.82, 0.60, 0.35, 0.20, 0.15])
        sugarcane_ndvi = np.array([0.30, 0.42, 0.55, 0.68, 0.75, 0.78, 0.76, 0.72, 0.65, 0.50])
        other_ndvi = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.38, 0.32, 0.28, 0.22, 0.18])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_steps, paddy_ndvi, 'o-', label='Paddy (Rapid peak & harvest drop)', color='#10B981', linewidth=2.5)
        ax.plot(time_steps, sugarcane_ndvi, 's-', label='Sugarcane (Extended high greenness)', color='#3B82F6', linewidth=2.5)
        ax.plot(time_steps, other_ndvi, '^-', label='Other Crops (Moderate growth)', color='#F59E0B', linewidth=2.5)
        
        ax.set_xlabel('Satellite Acquisition Time Steps', fontsize=12)
        ax.set_ylabel('NDVI Value', fontsize=12)
        ax.set_title('Crop Phenological Growth Trajectories (Sentinel-2)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=11)
        st.pyplot(fig)

        st.info(
            "**Key Phenological Insights:**\n\n"
            "- **Paddy**: Shows a sharp increase during the tillering/heading phase followed by a steep decrease during senescence/harvest.\n"
            "- **Sugarcane**: Exhibits a longer vegetation period with sustained high NDVI values due to extended crop growth cycle (~12-14 months).\n"
            "- **1D Inception Advantage**: The multi-scale 1D CNN captures both fast phenological shifts (Paddy) and prolonged greenness (Sugarcane)."
        )

    # 4. SATELLITE RASTER & GIS TOOL
    elif navigation == "🛠️ Satellite Raster & GIS Tool":
        st.markdown('<div class="main-header">🛠️ Geospatial Coordinate & Pixel Converter</div>', unsafe_allow_html=True)
        st.markdown("Tool for converting global Latitude / Longitude coordinates (EPSG:4326) into raster matrix indices.")

        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=30.9614, format="%.6f")
            lon = st.number_input("Longitude", value=76.4530, format="%.6f")
        with col2:
            red_val = st.slider("Sample Red Band (B6) Reflection", 0, 10000, 1200)
            nir_val = st.slider("Sample NIR Band (B8) Reflection", 0, 10000, 4800)

        ndvi_calc = calculate_ndvi(np.array([red_val]), np.array([nir_val]))[0]
        
        st.subheader("Calculated Spectral Feature")
        st.metric(label="Calculated NDVI", value=f"{ndvi_calc:.4f}")
        
        if ndvi_calc > 0.6:
            st.success("Dense Healthy Vegetation Detected (High likelihood of Paddy/Sugarcane at peak phase)")
        elif ndvi_calc > 0.3:
            st.info("Moderate Vegetation / Early Growth Phase")
        else:
            st.warning("Low Vegetation / Bare Soil / Water Body")


if __name__ == "__main__":
    main()

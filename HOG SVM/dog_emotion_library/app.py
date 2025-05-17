import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Dog Emotion Classifier",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown("""
<style>
    .title {
        text-align: center;
        color: #2e6e80;
    }
    .subtitle {
        text-align: center;
        color: #4e8d9c;
    }
    .prediction {
        font-size: 26px;
        font-weight: bold;
        text-align: center;
    }
    .confidence {
        font-size: 20px;
        text-align: center;
    }
    .stApp {
        background-color: #f5f7f9;
    }
    .upload-box {
        border: 2px dashed #aaa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Function to load model
@st.cache_resource
def load_model(model_path="dog_emotion_model_fast.pkl"):
    """Load the trained model from pkl file"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found in the current directory.")
        st.info(f"Current working directory: {os.getcwd()}")
        st.info(f"Files in directory: {os.listdir('.')}")
        return None

# Function to process image and make prediction
def process_image_and_predict(image, model_data):
    """Process image and predict dog emotion"""
    start_time = time.time()
    
    # Convert streamlit uploaded image to cv2 format
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Preprocess image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 64))  # HOG-friendly size
    
    # Extract HOG parameters from model
    hog_params = model_data['hog_params']
    
    # Extract HOG features
    features = hog(
        resized,
        orientations=hog_params['orientations'],
        pixels_per_cell=hog_params['pixels_per_cell'],
        cells_per_block=hog_params['cells_per_block'],
        block_norm='L2-Hys',
        transform_sqrt=hog_params['transform_sqrt']
    )
    
    # Apply feature selection
    feature_selector = model_data['feature_selector']
    features_reduced = feature_selector.transform([features])
    
    # Predict
    model = model_data['model']
    prediction = model.predict(features_reduced)[0]
    
    # Get probabilities
    calibrated_model = model_data['calibrated_model']
    probabilities = calibrated_model.predict_proba(features_reduced)[0]
    classes = model_data['classes']
    prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
    
    processing_time = time.time() - start_time
    
    return {
        'original_img': img,
        'preprocessed_img': resized,
        'prediction': prediction,
        'probabilities': prob_dict,
        'processing_time': processing_time
    }

# Function to display result
def display_result(result):
    """Display the prediction result with visualizations"""
    col1, col2 = st.columns(2)
    
    # Display original image
    with col1:
        st.image(cv2.cvtColor(result['original_img'], cv2.COLOR_BGR2RGB), 
                caption="Original Image", use_column_width=True)
    
    # Display preprocessed image
    with col2:
        st.image(result['preprocessed_img'], caption="Preprocessed Image (HOG Input)", 
                use_column_width=True)
    
    # Display prediction
    st.markdown(f"<p class='prediction'>Predicted Emotion: {result['prediction'].upper()}</p>", 
               unsafe_allow_html=True)
    st.markdown(f"<p class='confidence'>Processing Time: {result['processing_time']:.3f} seconds</p>", 
               unsafe_allow_html=True)
    
    # Create confidence chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    emotions = list(result['probabilities'].keys())
    confidence_scores = list(result['probabilities'].values())
    
    # Create bar colors (highlight the predicted one)
    colors = ['lightblue'] * len(emotions)
    predicted_idx = emotions.index(result['prediction'])
    colors[predicted_idx] = 'orange'
    
    # Plot bars
    bars = ax.bar(emotions, confidence_scores, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Customize plot
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('Emotion Confidence Scores', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Render chart
    st.pyplot(fig)

# Main app
def main():
    # App header
    st.markdown("<h1 class='title'>🐶 Dog Emotion Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='subtitle'>Upload a dog image to detect its emotion</h3>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses HOG (Histogram of Oriented Gradients) features "
        "and a Support Vector Machine (SVM) classifier to detect dog emotions. "
        "The model can classify 4 emotions: happy, sad, angry, and relaxed."
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        "1. Upload a dog image (jpg, jpeg, or png)\n"
        "2. The app will process the image and display the predicted emotion\n"
        "3. Confidence scores are shown as a graph"
    )
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.warning("Failed to load model. Please check if the model file exists in the correct location.")
        return
    
    # File uploader
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a dog image...", 
                                    type=["jpg", "jpeg", "png"],
                                    help="Upload a clear image of a dog face")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display loading spinner during processing
        with st.spinner("Processing image..."):
            # Make prediction
            result = process_image_and_predict(uploaded_file, model_data)
            
            # Reset file position for displaying
            uploaded_file.seek(0)
            
            # Display result
            display_result(result)
            
            # Display confidence scores in table format
            with st.expander("View Detailed Confidence Scores"):
                confidence_df = pd.DataFrame({
                    'Emotion': result['probabilities'].keys(),
                    'Confidence': result['probabilities'].values()
                })
                confidence_df = confidence_df.sort_values('Confidence', ascending=False)
                st.dataframe(confidence_df, use_container_width=True)

# For importing pandas
import pandas as pd

if __name__ == "__main__":
    main()
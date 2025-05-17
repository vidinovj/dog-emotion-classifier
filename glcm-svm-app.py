import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
import io
import matplotlib.pyplot as plt

# Set up page config
st.set_page_config(page_title="Dog Emotion Classifier", layout="wide")

# Define constants
EMOTIONS = ['Sad', 'Angry', 'Happy', 'Relaxed']
MODEL_PATH = 'glcm_svm_model.pkl'

# GLCM Feature Extraction Functions
def create_glcm_from_scratch(image, levels=8, distance=1, angles=None):
    """
    Compute Gray Level Co-occurrence Matrix (GLCM) from scratch using only NumPy
    """
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
    
    # Quantize the image to reduce gray levels
    bins = np.linspace(0, 255, levels+1)
    
    # Manual quantization
    quantized = np.zeros_like(image, dtype=np.int32)
    for i in range(1, len(bins)):
        quantized += (image >= bins[i-1]) & (image < bins[i]) * (i-1)
    # Handle edge case for max value
    quantized += (image == 255) * (levels-1)
    
    # Initialize GLCM
    glcm = np.zeros((levels, levels))
    
    h, w = quantized.shape
    
    # For each angle
    for angle in angles:
        dx = int(round(distance * np.cos(angle)))
        dy = int(round(distance * np.sin(angle)))
        
        temp_glcm = np.zeros((levels, levels))
        
        # Define the valid regions for reference and shifted pixels
        if dy >= 0:
            row_start_ref, row_end_ref = 0, h - dy
            row_start_shift, row_end_shift = dy, h
        else:
            row_start_ref, row_end_ref = -dy, h
            row_start_shift, row_end_shift = 0, h + dy
            
        if dx >= 0:
            col_start_ref, col_end_ref = 0, w - dx
            col_start_shift, col_end_shift = dx, w
        else:
            col_start_ref, col_end_ref = -dx, w
            col_start_shift, col_end_shift = 0, w + dx
        
        # Extract reference and shifted regions
        ref_region = quantized[row_start_ref:row_end_ref, col_start_ref:col_end_ref]
        shifted_region = quantized[row_start_shift:row_end_shift, col_start_shift:col_end_shift]
        
        # Manual GLCM computation
        for i in range(ref_region.shape[0]):
            for j in range(ref_region.shape[1]):
                ref_val = ref_region[i, j]
                shift_val = shifted_region[i, j]
                temp_glcm[ref_val, shift_val] += 1
        
        # Add to the accumulated GLCM
        glcm += temp_glcm
    
    # Ensure symmetry (i,j) = (j,i)
    glcm = glcm + glcm.T
    
    # Subtract the double-counted diagonal
    for i in range(levels):
        glcm[i, i] = glcm[i, i] / 2
    
    # Normalize GLCM
    glcm_sum = np.sum(glcm)
    if glcm_sum > 0:
        glcm = glcm / glcm_sum
        
    return glcm

def extract_glcm_features_from_scratch(glcm):
    """
    Extract Haralick texture features from GLCM using only NumPy
    """
    eps = 1e-10  # Small value to avoid division by zero
    
    # Get dimensions
    levels = glcm.shape[0]
    
    # Create indices manually
    i_indices = np.zeros((levels, levels))
    j_indices = np.zeros((levels, levels))
    
    for i in range(levels):
        for j in range(levels):
            i_indices[i, j] = i
            j_indices[i, j] = j
    
    # Marginal probabilities
    px = np.sum(glcm, axis=1)
    py = np.sum(glcm, axis=0)
    
    # Mean and standard deviations
    mu_x = 0
    mu_y = 0
    for i in range(levels):
        mu_x += i * px[i]
        mu_y += i * py[i]
    
    sigma_x = 0
    sigma_y = 0
    for i in range(levels):
        sigma_x += px[i] * ((i - mu_x) ** 2)
        sigma_y += py[i] * ((i - mu_y) ** 2)
    
    sigma_x = np.sqrt(sigma_x)
    sigma_y = np.sqrt(sigma_y)
    
    # 1. Angular Second Moment (Energy)
    asm = 0
    for i in range(levels):
        for j in range(levels):
            asm += glcm[i, j] ** 2
    energy = np.sqrt(asm)
    
    # 2. Contrast
    contrast = 0
    for i in range(levels):
        for j in range(levels):
            contrast += glcm[i, j] * ((i - j) ** 2)
    
    # 3. Correlation
    correlation = 0
    if sigma_x > eps and sigma_y > eps:
        for i in range(levels):
            for j in range(levels):
                correlation += glcm[i, j] * ((i - mu_x) * (j - mu_y)) / (sigma_x * sigma_y)
    
    # 4. Homogeneity (Inverse Difference Moment)
    homogeneity = 0
    for i in range(levels):
        for j in range(levels):
            homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
    
    # 5. Entropy
    entropy = 0
    for i in range(levels):
        for j in range(levels):
            if glcm[i, j] > 0:  # Avoid log(0)
                entropy -= glcm[i, j] * np.log2(glcm[i, j])
    
    # 6. Dissimilarity
    dissimilarity = 0
    for i in range(levels):
        for j in range(levels):
            dissimilarity += glcm[i, j] * abs(i - j)
    
    return np.array([energy, contrast, correlation, homogeneity, entropy, dissimilarity])

def extract_glcm_regional_features_from_scratch(image, num_regions=4, levels=8, distances=[1]):
    """
    Extract GLCM features from multiple regions of an image using only NumPy
    """
    h, w = image.shape
    grid_size = int(np.sqrt(num_regions))
    region_h = h // grid_size
    region_w = w // grid_size

    all_features = []

    # Global image statistics to complement GLCM
    all_features.append(np.mean(image))
    all_features.append(np.std(image))

    # Histogram features (8 bins)
    hist = np.zeros(8)
    bin_width = 256 // 8

    # Manual histogram computation
    for i in range(8):
        lower = i * bin_width
        upper = (i + 1) * bin_width if i < 7 else 256
        hist[i] = np.sum((image >= lower) & (image < upper))

    # Normalize histogram
    hist = hist / np.sum(hist)
    all_features.extend(hist)

    # Process each region for specified distances
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract region
            start_row = i * region_h
            end_row = (i + 1) * region_h
            start_col = j * region_w
            end_col = (j + 1) * region_w

            region = image[start_row:end_row, start_col:end_col]

            # Skip very small regions
            if region.shape[0] < 5 or region.shape[1] < 5:
                continue

            for distance in distances:
                glcm = create_glcm_from_scratch(region, levels=levels, distance=distance)
                features = extract_glcm_features_from_scratch(glcm)
                all_features.extend(features)

    return np.array(all_features)

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray_image = image.astype(np.uint8)
    
    # Resize to match expected input
    pil_image = Image.fromarray(gray_image)
    resized_image = np.array(pil_image.resize((128, 64)))
    
    return resized_image

# DirectMulticlassSVM class definition for model deserialization
class DirectMulticlassSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.001, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.binary_classifiers = {}
    
    def fit(self, X, y):
        self.classes = np.unique(y)
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        votes = np.zeros((n_samples, n_classes))
        
        # Collect votes from each binary classifier
        for (i, j), (w, b) in self.binary_classifiers.items():
            decisions = np.dot(X, w) + b
            votes[decisions > 0, i] += 1
            votes[decisions <= 0, j] += 1
        
        # Return class with the most votes
        return np.argmax(votes, axis=1), votes

# Load model function with improved error handling
@st.cache_resource
def load_model(model_path):
    """Load model with better error handling and caching"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error(f"Error: File '{model_path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {type(e).__name__}: {e}")
        return None

# Feature extraction and prediction
def predict_emotion(model_data, image):
    """Extract features and predict emotion"""
    # Get feature dimension from model
    feature_mean = model_data['feature_mean']
    feature_dim = feature_mean.shape[0]
    
    # Extract features with the correct parameters
    features = extract_glcm_regional_features_from_scratch(
        image, num_regions=4, levels=16, distances=[1, 2, 3]
    )
    
    # Check if feature dimensions match
    if features.shape[0] != feature_dim:
        st.warning(f"Feature dimension mismatch: Got {features.shape[0]}, expected {feature_dim}. Adjusting...")
        
        # Pad or truncate features to match expected dimension
        adjusted_features = np.zeros(feature_dim)
        min_dim = min(features.shape[0], feature_dim)
        adjusted_features[:min_dim] = features[:min_dim]
        
        # If we need to pad, add small random values
        if min_dim < feature_dim:
            adjusted_features[min_dim:] = np.random.rand(feature_dim - min_dim) * 0.001
        
        features = adjusted_features
    
    # Normalize features
    # First apply Z-score normalization
    mean = model_data['feature_mean']
    std = model_data['feature_std']
    std[std < 1e-10] = 1.0  # Avoid division by zero
    features_normalized = (features - mean) / std
    
    # Then apply min-max to the z-scored data
    min_vals = model_data['feature_min']
    max_vals = model_data['feature_max']
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-10] = 1.0  # Avoid division by zero
    features_normalized = (features_normalized - min_vals) / range_vals
    
    # Make prediction
    model = model_data['model']
    prediction, votes = model.predict(features_normalized.reshape(1, -1))
    
    # Normalize votes to get confidence
    vote_sums = np.sum(votes, axis=1, keepdims=True)
    confidence = votes / vote_sums if np.any(vote_sums > 0) else votes
    
    return prediction[0], confidence[0]

# Main app
def main():
    st.title("Dog Emotion Classifier using GLCM-SVM")
    st.write("""
    This app detects dog emotions using Gray Level Co-occurrence Matrix (GLCM) 
    texture features and Support Vector Machines (SVM).
    
    Upload a dog photo to classify its emotion as Sad, Angry, Happy, or Relaxed.
    """)
    
    # Sidebar information
    st.sidebar.header("About")
    st.sidebar.write("""
    This model uses texture analysis to detect dog emotions.
    
    **Techniques Used:**
    - GLCM Texture Features
    - Edge Orientation Histograms
    - SVM Classification
    
    **Accuracy:** ~33% across all four emotion classes
    """)
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("Debug Mode")
    
    # Try to load the model
    model_data = load_model(MODEL_PATH)
    model_loaded = model_data is not None
    
    if model_loaded:
        st.success("Model loaded successfully!")
    else:
        st.error("Failed to load model. Please check if the model file exists.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and model_loaded:
        # Display the image
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Preprocess image
            img_array = np.array(image)
            processed_img = preprocess_image(img_array)
            
            # Make prediction
            with st.spinner('Analyzing image...'):
                prediction, confidence = predict_emotion(model_data, processed_img)
            
            # Display results
            st.subheader("Prediction Result")
            st.write(f"The dog appears to be: **{EMOTIONS[prediction]}**")
            
            # Display confidence
            st.subheader("Confidence Levels")
            
            # Create a horizontal bar chart for confidence
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(EMOTIONS, confidence, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_position = width + 0.01
                ax.text(label_position, bar.get_y() + bar.get_height()/2, 
                        f'{confidence[i]:.1%}', va='center')
            
            ax.set_xlim(0, max(confidence) * 1.2)  # Give some space for the labels
            ax.set_xlabel('Confidence')
            ax.set_title('Emotion Prediction Confidence')
            st.pyplot(fig)
            
            # Also show as text for accessibility
            for i, emotion in enumerate(EMOTIONS):
                # Create bar using Unicode block character
                bar_length = int(40 * float(confidence[i]))
                bar = "█" * bar_length
                # Display emotion name and bar
                st.write(f"{emotion}: {confidence[i]:.2f}  {bar}")
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            if debug_mode:
                import traceback
                st.code(traceback.format_exc())
    
    elif uploaded_file is not None and not model_loaded:
        st.error("Cannot analyze image because the model failed to load.")
    
    # Add debug mode tools
    if debug_mode and uploaded_file is not None and model_loaded:
        st.subheader("Debug Information")
        
        # Show processed image
        st.write("Processed Image (128x64 grayscale):")
        st.image(processed_img, caption="Processed Image", width=128*2)
        
        # Show feature extraction process
        if st.checkbox("Show GLCM Features"):
            features = extract_glcm_regional_features_from_scratch(processed_img, num_regions=4, levels=16, distances=[1, 2, 3])
            st.write(f"Feature vector length: {len(features)}")
            st.write("First 10 feature values:", features[:10])
            
            # Check feature mismatch with model
            expected_dim = model_data['feature_mean'].shape[0]
            st.write(f"Model expects {expected_dim} features, got {len(features)} features")
            
            # Feature histogram
            fig, ax = plt.subplots()
            ax.hist(features, bins=20)
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Feature Value Distribution")
            st.pyplot(fig)
        
        # Show model information
        if st.checkbox("Show Model Information"):
            st.write("Model parameters:")
            st.write(f"Learning rate: {model_data['model'].lr}")
            st.write(f"Lambda: {model_data['model'].lambda_param}")
            st.write(f"Iterations: {model_data['model'].n_iters}")
            st.write(f"Number of binary classifiers: {len(model_data['model'].binary_classifiers)}")
            
            # Show a sample of the classifier weights
            st.write("Sample of classifier weights:")
            for pair, (w, b) in list(model_data['model'].binary_classifiers.items())[:1]:
                st.write(f"Classifier {pair}:")
                st.write(f"  Bias: {b}")
                st.write(f"  First 5 weights: {w[:5]}")

if __name__ == "__main__":
    main()

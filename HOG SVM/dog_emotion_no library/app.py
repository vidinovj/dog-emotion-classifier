import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
import io
import matplotlib.pyplot as plt
import math

# Set up page config
st.set_page_config(page_title="HOG Dog Emotion Classifier", layout="wide")

# Define constants
EMOTIONS = ['angry', 'happy', 'relaxed', 'sad']
MODEL_PATH = 'models/dog_emotion_svm_final.pkl'
METADATA_PATH = 'models/dog_emotion_svm_metadata.pkl'

# HOG Feature Extraction Functions
def compute_gradients(image):
    """Menghitung gradien x dan y dari gambar"""
    height = len(image)
    width = len(image[0])
    gradient_x = [[0 for _ in range(width)] for _ in range(height)]
    gradient_y = [[0 for _ in range(width)] for _ in range(height)]
    
    # Hitung gradien untuk setiap pixel (kecuali border)
    for y in range(1, height-1):
        for x in range(1, width-1):
            gradient_x[y][x] = image[y][x+1] - image[y][x-1]
            gradient_y[y][x] = image[y+1][x] - image[y-1][x]
    
    return gradient_x, gradient_y

def compute_hog_features(image, cell_size=8, block_size=2, num_bins=9):
    """Ekstraksi fitur HOG"""
    height = len(image)
    width = len(image[0])
    
    # Jumlah cell
    cells_y = height // cell_size
    cells_x = width // cell_size
    
    # Hitung gradien
    gradient_x, gradient_y = compute_gradients(image)
    
    # Inisialisasi histogram untuk setiap cell
    cell_histograms = [[None for _ in range(cells_x)] for _ in range(cells_y)]
    
    # Hitung histogram orientasi untuk setiap cell
    for cell_y in range(cells_y):
        for cell_x in range(cells_x):
            histogram = [0] * num_bins
            
            # Loop melalui pixel dalam cell
            for y in range(cell_y * cell_size, (cell_y + 1) * cell_size):
                for x in range(cell_x * cell_size, (cell_x + 1) * cell_size):
                    if y < height and x < width:
                        # Hitung magnitude dan orientasi
                        gx = gradient_x[y][x]
                        gy = gradient_y[y][x]
                        magnitude = math.sqrt(gx*gx + gy*gy)
                        
                        # Orientasi dalam derajat (0-180)
                        if gx == 0 and gy == 0:
                            orientation = 0
                        else:
                            orientation = (math.degrees(math.atan2(gy, gx)) + 180) % 180
                        
                        # Distribusi ke bin
                        bin_idx = int(orientation / (180 / num_bins))
                        if bin_idx >= num_bins:
                            bin_idx = num_bins - 1
                            
                        histogram[bin_idx] += magnitude
            
            cell_histograms[cell_y][cell_x] = histogram
    
    # Normalisasi blok dan gabungkan fitur
    features = []
    for block_y in range(cells_y - block_size + 1):
        for block_x in range(cells_x - block_size + 1):
            block_features = []
            
            # Kumpulkan histogram untuk block
            for cell_y in range(block_y, block_y + block_size):
                for cell_x in range(block_x, block_x + block_size):
                    block_features.extend(cell_histograms[cell_y][cell_x])
            
            # Normalisasi L2
            sum_squares = sum(x*x for x in block_features) + 1e-6  # Epsilon untuk stabilitas
            normalized = [x / math.sqrt(sum_squares) for x in block_features]
            features.extend(normalized)
    
    return features

# Image preprocessing functions
def gaussian_blur(image_data, sigma=0.8):
    """Gaussian blur untuk mengurangi noise"""
    height = len(image_data)
    width = len(image_data[0])
    blurred = [[0 for _ in range(width)] for _ in range(height)]
    
    # Size kernel berdasarkan sigma (biasanya 3*sigma)
    kernel_size = max(3, int(2 * math.ceil(3 * sigma) + 1))
    if kernel_size % 2 == 0:  # Pastikan ukuran ganjil
        kernel_size += 1
    
    kernel_radius = kernel_size // 2
    
    # Buat kernel Gaussian
    kernel = []
    kernel_sum = 0
    for i in range(kernel_size):
        kernel_row = []
        for j in range(kernel_size):
            x = i - kernel_radius
            y = j - kernel_radius
            # Rumus Gaussian 2D
            value = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
            kernel_row.append(value)
            kernel_sum += value
        kernel.append(kernel_row)
    
    # Normalisasi kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] /= kernel_sum
    
    # Apply kernel using padding
    padded = [[0 for _ in range(width + 2*kernel_radius)] for _ in range(height + 2*kernel_radius)]
    
    # Padding dengan duplikasi tepi
    for i in range(height):
        for j in range(width):
            padded[i+kernel_radius][j+kernel_radius] = image_data[i][j]
    
    # Fill padding edges
    for i in range(kernel_radius):
        for j in range(width + 2*kernel_radius):
            padded[i][j] = padded[kernel_radius][j]  # Top padding
            padded[height+kernel_radius+i][j] = padded[height+kernel_radius-1][j]  # Bottom padding
    
    for i in range(height + 2*kernel_radius):
        for j in range(kernel_radius):
            padded[i][j] = padded[i][kernel_radius]  # Left padding
            padded[i][width+kernel_radius+j] = padded[i][width+kernel_radius-1]  # Right padding
    
    # Konvolusi
    for i in range(height):
        for j in range(width):
            weighted_sum = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    weighted_sum += padded[i+ki][j+kj] * kernel[ki][kj]
            blurred[i][j] = int(weighted_sum)
    
    return blurred

def histogram_equalization(image_data):
    """Meningkatkan kontras gambar menggunakan histogram equalization"""
    height = len(image_data)
    width = len(image_data[0])
    equalized = [[0 for _ in range(width)] for _ in range(height)]
    
    # Hitung histogram
    histogram = [0] * 256
    for i in range(height):
        for j in range(width):
            histogram[image_data[i][j]] += 1
    
    # Hitung histogram kumulatif
    cum_hist = [0] * 256
    cum_hist[0] = histogram[0]
    for i in range(1, 256):
        cum_hist[i] = cum_hist[i-1] + histogram[i]
    
    # Normalisasi
    total_pixels = height * width
    norm_cum_hist = [0] * 256
    for i in range(256):
        norm_cum_hist[i] = int(cum_hist[i] * 255 / total_pixels)
    
    # Terapkan ekualisasi
    for i in range(height):
        for j in range(width):
            equalized[i][j] = norm_cum_hist[image_data[i][j]]
    
    return equalized

def enhance_edges(image_data, strength=0.3):
    """Mempertajam tepi gambar untuk HOG yang lebih menonjol"""
    height = len(image_data)
    width = len(image_data[0])
    enhanced = [[0 for _ in range(width)] for _ in range(height)]
    
    # Copy border pixels as is
    for i in range(height):
        enhanced[i][0] = image_data[i][0]
        enhanced[i][width-1] = image_data[i][width-1]
    
    for j in range(width):
        enhanced[0][j] = image_data[0][j]
        enhanced[height-1][j] = image_data[height-1][j]
    
    # Sobel operator for edge detection
    for y in range(1, height-1):
        for x in range(1, width-1):
            # Sobel horizontal
            gx = (image_data[y-1][x+1] + 2*image_data[y][x+1] + image_data[y+1][x+1]) - \
                 (image_data[y-1][x-1] + 2*image_data[y][x-1] + image_data[y+1][x-1])
            
            # Sobel vertical
            gy = (image_data[y+1][x-1] + 2*image_data[y+1][x] + image_data[y+1][x+1]) - \
                 (image_data[y-1][x-1] + 2*image_data[y-1][x] + image_data[y-1][x+1])
            
            # Calculate magnitude
            edge_strength = min(255, int(math.sqrt(gx*gx + gy*gy)))
            
            # Enhance original image with edge information
            enhanced[y][x] = min(255, image_data[y][x] + int(edge_strength * strength))
    
    return enhanced

def preprocess_image(image):
    """Preprocess image for model input with enhanced features"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray_image = image.astype(np.uint8)
    
    # Resize to match expected input (128x64)
    pil_image = Image.fromarray(gray_image)
    resized_image = pil_image.resize((128, 64))
    
    # Convert to 2D list for our HOG function
    pixel_data = []
    for y_coord in range(64):
        row_data = []
        for x_coord in range(128):
            pixel = resized_image.getpixel((x_coord, y_coord))
            row_data.append(pixel)
        pixel_data.append(row_data)
    
    # Apply optimizations
    blurred = gaussian_blur(pixel_data, sigma=0.8)
    equalized = histogram_equalization(blurred)
    enhanced = enhance_edges(equalized, strength=0.2)
    
    return enhanced

# LinearSVM class for model deserialization
class LinearSVM:
    def __init__(self, C=1.0):
        self.C = C
        self.weights = {}
        self.bias = 0
        self.classes = None
        self.feature_selection = None
    
    def _linear_kernel(self, x1, x2):
        if self.feature_selection is not None:
            return sum(x1[i] * x2[i] for i in self.feature_selection)
        return sum(a * b for a, b in zip(x1, x2))
    
    def predict(self, X):
        predictions = []
        confidences = []
        
        for x in X:
            scores = {}
            for c in self.classes:
                if self.feature_selection is None:
                    scores[c] = sum(w_i * x_i for w_i, x_i in zip(self.weights[c], x))
                else:
                    scores[c] = sum(self.weights[c][i] * x[self.feature_selection[i]] 
                                  for i in range(len(self.feature_selection)))
            
            # Get class with highest score
            pred_class = max(scores, key=scores.get)
            predictions.append(pred_class)
            
            # Convert scores to confidence with softmax
            score_values = list(scores.values())
            max_score = max(score_values)
            exp_scores = [math.exp(s - max_score) for s in score_values]  # Numerically stable
            sum_exp = sum(exp_scores)
            softmax_scores = [e / sum_exp for e in exp_scores]
            
            # Map back to emotion classes
            conf_dict = {}
            for i, c in enumerate(self.classes):
                conf_dict[c] = softmax_scores[i]
                
            confidences.append(conf_dict)
        
        return predictions, confidences

# Load model function with improved error handling
@st.cache_resource
def load_model(model_path, metadata_path):
    """Load model with better error handling and caching"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        return {'model': model, 'metadata': metadata}
    except FileNotFoundError:
        st.error(f"Error: Model files not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {type(e).__name__}: {e}")
        return None

# Feature extraction and prediction
def predict_emotion(model_data, image):
    """Extract features and predict emotion"""
    # Extract HOG features
    features = compute_hog_features(image)
    
    # Make prediction
    model = model_data['model']
    predictions, confidences = model.predict([features])
    
    # Get confidence values in dictionary form
    confidence_dict = confidences[0]
    
    return predictions[0], confidence_dict

# Main app
def main():
    st.title("Dog Emotion Classifier using HOG-SVM")
    st.write("""
    This app detects dog emotions using Histogram of Oriented Gradients (HOG) 
    features and Support Vector Machines (SVM).
    
    Upload a dog photo to classify its emotion as Angry, Happy, Relaxed, or Sad.
    """)
    
    # Sidebar information
    st.sidebar.header("About")
    st.sidebar.write("""
    This model uses HOG feature extraction to detect dog emotions.
    
    **Techniques Used:**
    - Histogram of Oriented Gradients (HOG)
    - Linear Support Vector Machine (SVM)
    - Image enhancement (Gaussian blur, Histogram equalization, Edge enhancement)
    
    **Model Parameters:**
    - Kernel: Linear
    - C: Optimized through validation
    """)
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("Debug Mode")
    
    # Try to load the model
    model_data = load_model(MODEL_PATH, METADATA_PATH)
    model_loaded = model_data is not None
    
    if model_loaded:
        st.success("Model loaded successfully!")
        # Show metadata if available
        metadata = model_data['metadata']
        if debug_mode and metadata:
            st.sidebar.subheader("Model Metadata")
            st.sidebar.write(f"Training samples: {metadata.get('training_samples', 'N/A')}")
            st.sidebar.write(f"Test accuracy: {metadata.get('test_accuracy', 0):.3f}")
            st.sidebar.write(f"Parameter C: {metadata.get('parameters', {}).get('C', 'N/A')}")
    else:
        st.error("Failed to load model. Please check if the model files exist.")
    
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
                prediction, confidence_dict = predict_emotion(model_data, processed_img)
            
            # Display results
            st.subheader("Prediction Result")
            st.write(f"The dog appears to be: **{prediction.upper()}**")
            
            # Display confidence
            st.subheader("Confidence Levels")
            
            # Create a horizontal bar chart for confidence
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Colors for each emotion
            colors = {'angry': '#ff9999', 'happy': '#99ff99', 'relaxed': '#66b3ff', 'sad': '#ffcc99'}
            
            # Sort emotions by confidence values
            sorted_emotions = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
            emotions = [item[0] for item in sorted_emotions]
            confidence_values = [item[1] for item in sorted_emotions]
            
            # Create the bar chart
            bars = ax.barh(emotions, confidence_values, color=[colors.get(e, '#cccccc') for e in emotions])
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_position = width + 0.01
                ax.text(label_position, bar.get_y() + bar.get_height()/2, 
                        f'{confidence_values[i]:.1%}', va='center')
            
            ax.set_xlim(0, max(confidence_values) * 1.2)  # Give some space for the labels
            ax.set_xlabel('Confidence')
            ax.set_title('Emotion Prediction Confidence')
            st.pyplot(fig)
            
            # Also show as text for accessibility
            for emotion, conf in sorted_emotions:
                # Create bar using Unicode block character
                bar_length = int(40 * float(conf))
                bar = "█" * bar_length
                # Display emotion name and bar
                st.write(f"{emotion.upper()}: {conf:.2f}  {bar}")
            
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
        processed_array = np.array(processed_img)
        st.image(processed_array, caption="Processed Image", width=256)
        
        # Show HOG features
        if st.checkbox("Show HOG Features"):
            features = compute_hog_features(processed_img)
            st.write(f"Feature vector length: {len(features)}")
            st.write("First 10 feature values:", features[:10])
            
            # Feature histogram
            fig, ax = plt.subplots()
            ax.hist(features, bins=20)
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Frequency")
            ax.set_title("HOG Feature Value Distribution")
            st.pyplot(fig)
        
        # Show model information
        if st.checkbox("Show Model Information"):
            model = model_data['model']
            st.write("Model parameters:")
            st.write(f"C: {model.C}")
            st.write(f"Classes: {model.classes}")
            
            if model.feature_selection:
                st.write(f"Using feature selection: {len(model.feature_selection)} features selected")
            else:
                st.write("Using all features (no feature selection)")
            
            # Show class accuracies if available in metadata
            if 'class_accuracy' in model_data['metadata']:
                st.write("Per-class accuracies:")
                class_acc = model_data['metadata']['class_accuracy']
                for cls, acc in class_acc.items():
                    st.write(f"  - {cls}: {acc:.3f}")

if __name__ == "__main__":
    main()
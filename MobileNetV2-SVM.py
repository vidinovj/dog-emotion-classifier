import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm
import joblib

# Check if GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Function to load and preprocess images for CNN feature extraction
def load_and_preprocess_for_cnn(img_path):
    """Load and preprocess image for CNN feature extraction"""
    try:
        # For CNNs, we need color images in RGB format
        # Load as RGB with target size required by the model
        img = load_img(img_path, target_size=(224, 224))  # Standard size for many CNNs
        # Convert to array
        img_array = img_to_array(img)
        # Expand dimensions for batch processing
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocess for the specific model
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Function to extract features using a pre-trained CNN
def extract_cnn_features(base_model, image_paths, batch_size=16):
    """Extract features using a pre-trained CNN"""
    features = []
    valid_paths = []

    # Process images in batches for efficiency
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CNN features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_batch_paths = []

        for img_path in batch_paths:
            img = load_and_preprocess_for_cnn(img_path)
            if img is not None:
                batch_images.append(img[0])  # Remove batch dimension
                valid_batch_paths.append(img_path)

        if batch_images:
            batch_images = np.array(batch_images)
            batch_features = base_model.predict(batch_images, verbose=0)

            # Flatten features
            batch_features = batch_features.reshape(batch_features.shape[0], -1)
            features.extend(batch_features)
            valid_paths.extend(valid_batch_paths)

    return np.array(features), valid_paths

# Fast model creation function
def create_model_from_best_params(base_dir, num_samples_per_class=30, random_seed=42):
    """
    Create and save a model using the best parameters from previous training,
    using a small subset of data for quick training.
    """
    print("Creating model with best parameters from previous training...")
    start_time = time.time()

    # Set paths and categories
    labels_path = os.path.join(base_dir, "labels.csv")
    categories = ['angry', 'happy', 'relaxed', 'sad']

    # Load labels
    try:
        labels_df = pd.read_csv(labels_path)
        print(f"Loaded labels dataframe with shape: {labels_df.shape}")
        print(labels_df.head())
    except Exception as e:
        print(f"Error loading labels: {e}")
        print("Creating a simple dataframe instead...")
        # Create a simple dataframe if the file doesn't exist
        # List all files in each category directory
        rows = []
        for i, category in enumerate(categories):
            category_dir = os.path.join(base_dir, category.lower())
            if os.path.exists(category_dir):
                files = os.listdir(category_dir)
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        rows.append({'filename': file, 'label': category})

        labels_df = pd.DataFrame(rows)
        print(f"Created labels dataframe with shape: {labels_df.shape}")
        if len(labels_df) == 0:
            print("No images found. Check your directory structure.")
            return None, None

    # Create full paths to images
    image_paths = []
    labels = []

    # Get a balanced sample of images per class
    for category in categories:
        category_df = labels_df[labels_df['label'].str.lower() == category.lower()]
        if len(category_df) > 0:
            # Sample images
            sample_size = min(num_samples_per_class, len(category_df))
            category_sample = category_df.sample(sample_size, random_state=random_seed)

            for idx, row in category_sample.iterrows():
                filename = row['filename']
                category_folder = category.lower()
                img_path = os.path.join(base_dir, category_folder, filename)

                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(categories.index(category.lower()))

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    print(f"Total images sampled: {len(image_paths)}")
    print(f"Labels distribution: {np.bincount(labels)}")

    # Load pre-trained CNN
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    print("CNN model loaded for feature extraction")

    # Extract features
    print("Extracting features for sampled images...")
    X_features, valid_paths = extract_cnn_features(base_model, image_paths)

    # Get corresponding labels for valid paths
    valid_indices = [i for i, path in enumerate(image_paths) if path in valid_paths]
    valid_labels = labels[valid_indices]

    print(f"Features extracted: {X_features.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Create SVM with best parameters from previous output
    best_params = {
        'C': 10,
        'gamma': 'scale',
        'kernel': 'rbf',
        'probability': True
    }

    print(f"Training SVM with best parameters: {best_params}")
    model = SVC(**best_params)
    model.fit(X_scaled, valid_labels)

    # Check accuracy
    accuracy = model.score(X_scaled, valid_labels)
    print(f"Training accuracy: {accuracy:.4f}")

    # Save the model and scaler
    joblib.dump(model, 'dog_emotion_cnn_svm_model.pkl')
    joblib.dump(scaler, 'cnn_feature_scaler.pkl')
    print("Model saved as 'dog_emotion_cnn_svm_model.pkl'")
    print("Scaler saved as 'cnn_feature_scaler.pkl'")

    # Calculate execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time/60:.2f} minutes")

    return model, scaler

# Create a simpler version for directories that might not match the expected structure
def create_model_with_sample_images(sample_dir, num_samples=20):
    """Create a model using sample images in provided directory"""
    print(f"Creating model with sample images from {sample_dir}...")

    # Expected directory structure: sample_dir/category/image_files
    categories = ['angry', 'happy', 'relaxed', 'sad']

    # Gather image paths
    image_paths = []
    labels = []

    for label_idx, category in enumerate(categories):
        category_dir = os.path.join(sample_dir, category.lower())
        if os.path.exists(category_dir):
            files = [f for f in os.listdir(category_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Sample files if there are too many
            if len(files) > num_samples:
                files = np.random.choice(files, num_samples, replace=False)

            for file in files:
                image_paths.append(os.path.join(category_dir, file))
                labels.append(label_idx)

    if not image_paths:
        print("No images found in the expected directory structure.")
        return None, None

    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    print(f"Total images found: {len(image_paths)}")
    print(f"Labels distribution: {np.bincount(labels)}")

    # Load pre-trained CNN
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )

    # Extract features
    X_features, valid_paths = extract_cnn_features(base_model, image_paths)

    # Get corresponding labels for valid paths
    valid_indices = [i for i, path in enumerate(image_paths) if path in valid_paths]
    valid_labels = labels[valid_indices]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Create SVM with best parameters
    model = SVC(C=10, gamma='scale', kernel='rbf', probability=True)
    model.fit(X_scaled, valid_labels)

    # Save the model and scaler
    joblib.dump(model, 'dog_emotion_cnn_svm_model.pkl')
    joblib.dump(scaler, 'cnn_feature_scaler.pkl')

    print("Model saved successfully!")
    return model, scaler

# Function to create a very simple demo model when no real data is available
def create_demo_model():
    """Create a demo model with artificial data when no real images are available"""
    print("Creating a demo model with artificial data...")

    # Create feature extractor
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )

    # Number of samples and feature dimensions
    n_samples_per_class = 30
    feature_dim = 1280  # MobileNetV2 feature dimension

    # Generate artificial data for 4 emotions
    X_train = []
    y_train = []

    # Create distinct patterns for each class
    for class_idx in range(4):
        # Base features
        features = np.random.normal(
            loc=0.0,
            scale=0.1,
            size=(n_samples_per_class, feature_dim)
        )

        # Add class-specific patterns
        for i in range(n_samples_per_class):
            # Set different ranges of features to have distinct values for each class
            start_idx = class_idx * 300
            end_idx = start_idx + 300

            # Different mean for each class
            mean_val = 0.2 + class_idx * 0.2
            features[i, start_idx:end_idx] = np.random.normal(
                loc=mean_val,
                scale=0.05,
                size=300
            )

        X_train.append(features)
        y_train.extend([class_idx] * n_samples_per_class)

    # Combine data
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train classifier with best parameters
    model = SVC(C=10, gamma='scale', kernel='rbf', probability=True)
    model.fit(X_scaled, y_train)

    # Check accuracy
    accuracy = model.score(X_scaled, y_train)
    print(f"Training accuracy on artificial data: {accuracy:.4f}")

    # Save model and scaler
    joblib.dump(model, 'dog_emotion_cnn_svm_model.pkl')
    joblib.dump(scaler, 'cnn_feature_scaler.pkl')

    print("Demo model saved successfully!")
    return model, scaler

# Main function that tries different approaches
def main():
    # Define possible data locations
    possible_paths = [
        "/content/sample_data/Dog_Emotion",  # Colab default
        "/content/Dog_Emotion",              # Colab alternative
        "/content/dataset/Dog_Emotion",      # Another alternative
        "Dog_Emotion",                       # Relative path
        "./Dog_Emotion",                     # Another relative path
    ]

    # Try to find your dataset
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found dataset at: {path}")
            found_path = path
            break

    if found_path:
        # Try to create model with existing dataset
        model, scaler = create_model_from_best_params(found_path)

        if model is None:
            # Try alternative approach
            print("Trying alternative approach with sample directories...")
            model, scaler = create_model_with_sample_images(found_path)
    else:
        print("Could not find the Dog_Emotion dataset folder.")
        print("Checking for individual emotion folders...")

        # Check if individual emotion folders exist in current directory
        categories = ['angry', 'happy', 'relaxed', 'sad']
        if all(os.path.exists(cat) for cat in categories):
            print("Found emotion folders in current directory.")
            model, scaler = create_model_with_sample_images(".")
        else:
            print("No suitable dataset structure found.")
            # Create a demo model with artificial data as last resort
            model, scaler = create_demo_model()

    print("Model creation process complete!")

if __name__ == "__main__":
    main()

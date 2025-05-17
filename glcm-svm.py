import numpy as np

# GLCM Feature Extraction (No External Libraries)
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

def extract_glcm_batch_from_scratch(images, batch_size=10):
    """
    Extract GLCM features from a batch of images using only NumPy
    """
    all_features = []
    total = len(images)

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch = images[i:end]

        # Process batch
        batch_features = []
        for img in batch:
            # Apply optimized GLCM extraction
            features = extract_glcm_regional_features_from_scratch(
                img, num_regions=4, levels=16, distances=[1, 2, 3])
            batch_features.append(features)

        all_features.extend(batch_features)
        print(f"Processed {end}/{total} images")

    return np.array(all_features)

# Dataset Loading and Preprocessing
def load_dataset_with_sampling(root_dir, samples_per_class=None):
    """
    Load dataset from directory structure with optional sampling
    """
    import os
    from PIL import Image

    classes = ['sad', 'angry', 'happy', 'relaxed']
    images = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        files = os.listdir(class_dir)

        # Sample if needed
        if samples_per_class is not None and samples_per_class < len(files):
            import random
            files = random.sample(files, samples_per_class)

        for file_name in files:
            # Load image
            img_path = os.path.join(class_dir, file_name)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((128, 64))  # Resize as per project specs

            # Convert to numpy array
            img_array = np.array(img)

            images.append(img_array)
            labels.append(class_idx)

    return np.array(images), np.array(labels)

def split_dataset(images, labels, train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset into train, validation, and test sets
    """
    # Shuffle data
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    # Calculate split indices
    n_train = int(len(images) * train_ratio)
    n_val = int(len(images) * val_ratio)

    # Split data
    X_train = images[:n_train]
    y_train = labels[:n_train]

    X_val = images[n_train:n_train+n_val]
    y_val = labels[n_train:n_train+n_val]

    X_test = images[n_train+n_val:]
    y_test = labels[n_train+n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Multiclass SVM Implementation
class DirectMulticlassSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.001, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.binary_classifiers = {}

    def fit(self, X, y):
      self.classes = np.unique(y)
      n_classes = len(self.classes)

      # Create one-vs-one classifiers with better balance
      for i in range(n_classes):
          for j in range(i+1, n_classes):
              print(f"Training classifier for classes {i} vs {j}")

              # Get indices for class i and j
              i_indices = np.where(y == i)[0]
              j_indices = np.where(y == j)[0]

              # Balance with moderate augmentation for underrepresented classes
              min_samples = min(len(i_indices), len(j_indices))
              max_samples = min(min_samples * 1.2, max(len(i_indices), len(j_indices)))

              # Resample with balanced approach
              if len(i_indices) < len(j_indices):
                  i_indices = np.random.choice(i_indices, int(max_samples), replace=True)
                  j_indices = np.random.choice(j_indices, int(min_samples), replace=False)
              else:
                  i_indices = np.random.choice(i_indices, int(min_samples), replace=False)
                  j_indices = np.random.choice(j_indices, int(max_samples), replace=True)

              # Extract samples and labels
              X_ij = np.vstack((X[i_indices], X[j_indices]))
              y_ij = np.hstack((np.ones(len(i_indices)), -np.ones(len(j_indices))))

              # Better initialization - using mean of data points
              w = np.zeros(X.shape[1])
              b = 0

              # Improved learning process
              best_w = None
              best_b = None
              best_accuracy = 0

              # Multiple learning rates to find the best one
              learning_rates = [0.01, 0.005, 0.001]

              for lr in learning_rates:
                  w_temp = np.zeros(X.shape[1])
                  b_temp = 0

                  # Mini-batch gradient descent
                  batch_size = 32
                  for epoch in range(self.n_iters):
                      # Shuffle data
                      indices = np.random.permutation(len(X_ij))
                      X_ij_shuffled = X_ij[indices]
                      y_ij_shuffled = y_ij[indices]

                      # Mini-batch updates
                      for start_idx in range(0, len(X_ij), batch_size):
                          end_idx = min(start_idx + batch_size, len(X_ij))
                          X_batch = X_ij_shuffled[start_idx:end_idx]
                          y_batch = y_ij_shuffled[start_idx:end_idx]

                          grad_w = self.lambda_param * w_temp
                          grad_b = 0

                          for idx, x_sample in enumerate(X_batch):
                              margin = y_batch[idx] * (np.dot(x_sample, w_temp) + b_temp)

                              if margin < 1:
                                  grad_w -= y_batch[idx] * x_sample / len(X_batch)
                                  grad_b -= y_batch[idx] / len(X_batch)

                          # Update with current learning rate
                          w_temp -= lr * grad_w
                          b_temp -= lr * grad_b

                  # Evaluate this model
                  correct = 0
                  for idx, x_sample in enumerate(X_ij):
                      prediction = 1 if np.dot(x_sample, w_temp) + b_temp > 0 else -1
                      if prediction == y_ij[idx]:
                          correct += 1
                  accuracy = correct / len(X_ij)

                  # Save if best
                  if accuracy > best_accuracy:
                      best_accuracy = accuracy
                      best_w = w_temp.copy()
                      best_b = b_temp

              # Store best classifier parameters
              self.binary_classifiers[(i, j)] = (best_w, best_b)
              print(f" Binary accuracy ({i} vs {j}): {best_accuracy:.4f}")

    def predict(self, X):
      n_samples = X.shape[0]
      n_classes = len(self.classes)
      votes = np.zeros((n_samples, n_classes))

      # Collect raw decision values
      decision_values = {}
      for (i, j), (w, b) in self.binary_classifiers.items():
          decision_values[(i, j)] = np.dot(X, w) + b

      # More balanced class weights - gentle adjustments
      class_weights = np.array([1.15, 1.15, 0.95, 0.9])  # More balanced weights

      # Apply calibrated voting
      for (i, j), decisions in decision_values.items():
          # Use normalized decision values
          confidence = np.abs(decisions)
          max_conf = np.max(confidence) if np.max(confidence) > 0 else 1.0
          normalized_conf = confidence / max_conf

          # Weighted votes for class i
          votes[decisions > 0, i] += class_weights[i] * (1 + normalized_conf[decisions > 0] * 0.2)
          # Weighted votes for class j
          votes[decisions <= 0, j] += class_weights[j] * (1 + normalized_conf[decisions <= 0] * 0.2)

      # Print vote distributions for debugging
      print("\nAverage votes per class:")
      for c in range(n_classes):
          print(f" Class {c}: {np.mean(votes[:, c]):.2f}")

      # Return class with the most votes
      return np.argmax(votes, axis=1)

# Evaluation Metrics
def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    """
    n_classes = len(np.unique(y_true))

    # Initialize metrics
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)

    # Confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        confusion[y_true[i], y_pred[i]] += 1

    # Calculate per-class metrics
    for c in range(n_classes):
        # True positives
        tp = confusion[c, c]

        # False positives (sum of column c excluding true positives)
        fp = np.sum(confusion[:, c]) - tp

        # False negatives (sum of row c excluding true positives)
        fn = np.sum(confusion[c, :]) - tp

        # Calculate metrics
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0

    # Macro-averaged metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Accuracy
    accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion': confusion
    }

def print_metrics(metrics, class_names):
    """
    Print evaluation metrics
    """
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"Class {class_name}:")
        print(f"  Precision: {metrics['precision'][i]:.4f}")
        print(f"  Recall: {metrics['recall'][i]:.4f}")
        print(f"  F1-score: {metrics['f1'][i]:.4f}")

    print("\nMacro-averaged metrics:")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall: {metrics['macro_recall']:.4f}")
    print(f"F1-score: {metrics['macro_f1']:.4f}")

    print("\nConfusion Matrix:")
    confusion_str = str(metrics['confusion']).replace('[', '').replace(']', '')
    lines = confusion_str.split('\n')
    for i, line in enumerate(lines):
        if i < len(class_names):
            print(f"{line}  # {class_names[i]}")
        else:
            print(line)

# Main Function
def main():
    # Set random seed
    np.random.seed(42)

    # Check for existing GLCM feature files
    try:
        print("Checking for saved GLCM features...")
        X_train_features = np.load('glcm_train_features.npy')
        y_train = np.load('train_labels.npy')
        X_val_features = np.load('glcm_val_features.npy')
        y_val = np.load('val_labels.npy')
        X_test_features = np.load('glcm_test_features.npy')
        y_test = np.load('test_labels.npy')
        print("Loaded GLCM features from files")
    except:
        # Load existing image data if available
        try:
            print("Loading saved image data...")
            X_train = np.load('train_images.npy')
            y_train = np.load('train_labels.npy')
            X_val = np.load('val_images.npy')
            y_val = np.load('val_labels.npy')
            X_test = np.load('test_images.npy')
            y_test = np.load('test_labels.npy')
            print("Loaded image data from files")
        except:
            # Load from original dataset
            print("Loading dataset from raw images...")
            root_dir = '/content/dataset/Dog_Emotion'
            samples_per_class = 1000  # Using full dataset (1000 per class)
            images, labels = load_dataset_with_sampling(root_dir, samples_per_class)

            # Split dataset
            X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(images, labels)

            # Save image data
            np.save('train_images.npy', X_train)
            np.save('train_labels.npy', y_train)
            np.save('val_images.npy', X_val)
            np.save('val_labels.npy', y_val)
            np.save('test_images.npy', X_test)
            np.save('test_labels.npy', y_test)

        # Extract GLCM features
        print("Extracting GLCM features...")
        X_train_features = extract_glcm_batch_from_scratch(X_train, batch_size=10)
        X_val_features = extract_glcm_batch_from_scratch(X_val, batch_size=10)
        X_test_features = extract_glcm_batch_from_scratch(X_test, batch_size=10)

        # Save GLCM features
        np.save('glcm_train_features.npy', X_train_features)
        np.save('glcm_val_features.npy', X_val_features)
        np.save('glcm_test_features.npy', X_test_features)

        # ADD FEATURE SELECTION HERE - After features are loaded/extracted but before normalization
        print("Performing feature selection...")
        feature_variances = np.var(X_train_features, axis=0)
        # Keep only features with variance above threshold
        threshold = np.percentile(feature_variances, 10)  # Keep top 90% of features
        keep_indices = np.where(feature_variances > threshold)[0]
        X_train_features = X_train_features[:, keep_indices]
        X_val_features = X_val_features[:, keep_indices]
        X_test_features = X_test_features[:, keep_indices]
        print(f"Reduced from {len(feature_variances)} to {len(keep_indices)} features")

    # Normalize features
    mean = np.mean(X_train_features, axis=0)
    std = np.std(X_train_features, axis=0) 
    std[std < 1e-10] = 1.0
    X_train_features = (X_train_features - mean) / std
    X_val_features = (X_val_features - mean) / std
    X_test_features = (X_test_features - mean) / std

    # Then apply min-max to the z-scored data
    min_vals = np.min(X_train_features, axis=0)
    max_vals = np.max(X_train_features, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-10] = 1.0
    X_train_features = (X_train_features - min_vals) / range_vals
    X_val_features = (X_val_features - min_vals) / range_vals
    X_test_features = (X_test_features - min_vals) / range_vals

    # Print class distribution
    for dataset_name, y in [("Training", y_train), ("Validation", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n{dataset_name} class distribution:")
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} samples ({c/len(y)*100:.1f}%)")

    # Train model with fixed decision boundaries
    print("\nTraining direct multiclass SVM...")
    # Try a slightly larger lambda to help with generalization
    model = DirectMulticlassSVM(learning_rate=0.008, lambda_param=0.005, n_iters=1500)
    model.fit(X_train_features, y_train)

    # Validate
    print("\nValidating model...")
    y_val_pred = model.predict(X_val_features)
    val_accuracy = np.mean(y_val_pred == y_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Create validation confusion matrix
    val_confusion = np.zeros((4, 4), dtype=int)
    for i in range(len(y_val)):
        val_confusion[y_val[i], y_val_pred[i]] += 1

    print("Validation confusion matrix:")
    print(val_confusion)

    # Test evaluation
    print("\nEvaluating on test set...")
    y_test_pred = model.predict(X_test_features)

    # Calculate and print metrics
    metrics = calculate_metrics(y_test, y_test_pred)
    class_names = ['sad', 'angry', 'happy', 'relaxed']
    print_metrics(metrics, class_names)

    # Save model
    print("Saving final model...")
    import pickle
    with open('glcm_svm_model.pkl', 'wb') as f:
      pickle.dump({
          'model': model,
          'feature_mean': mean,
          'feature_std': std,
          'feature_min': min_vals,
          'feature_max': max_vals
      }, f)

    print("Model saved to glcm_svm_model.pkl")

if __name__ == "__main__":
    main()

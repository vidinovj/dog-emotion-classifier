import numpy as np

def create_glcm_from_scratch(image, levels=16, distance=1, angles=None):
    """
    Compute Gray Level Co-occurrence Matrix (GLCM) from scratch using only NumPy

    Parameters:
    -----------
    image : ndarray
        Input grayscale image
    levels : int
        Number of gray levels (reduces computation complexity)
    distance : int
        Distance between pixel pairs
    angles : list of floats
        Angles for co-occurrence in radians. Default [0, π/4, π/2, 3π/4]

    Returns:
    --------
    glcm : ndarray
        Normalized GLCM matrix averaged over all angles
    """
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°

    # Quantize the image to reduce gray levels (crucial for efficiency)
    # Create bins for quantization
    bins = np.linspace(0, 255, levels+1)

    # Manual quantization without using scipy
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

        # Manual computation with array slicing (no scipy.spatial.distance)
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

        # Manual GLCM computation using nested loops (no vectorization)
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

    Parameters:
    -----------
    glcm : ndarray
        Normalized GLCM matrix

    Returns:
    --------
    features : ndarray
        Array of GLCM features
    """
    eps = 1e-10  # Small value to avoid division by zero

    # Get dimensions
    levels = glcm.shape[0]

    # Create indices manually (no meshgrid)
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

def extract_glcm_regional_features_from_scratch(image, num_regions=4, levels=16, distances=[1, 2, 3]):
    """
    Enhanced GLCM feature extraction with orientation histograms
    """
    h, w = image.shape
    grid_size = int(np.sqrt(num_regions))
    region_h = h // grid_size
    region_w = w // grid_size

    all_features = []

    # Global image statistics (keep your existing code)
    all_features.append(np.mean(image))
    all_features.append(np.std(image))

    # Histogram features (keep your existing code)
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

    # NEW: Add orientation histogram for entire image
    # Compute image gradients
    gy = np.zeros_like(image, dtype=float)
    gx = np.zeros_like(image, dtype=float)

    # Compute x gradient (horizontal)
    for i in range(h):
        for j in range(1, w-1):
            gx[i, j] = float(image[i, j+1]) - float(image[i, j-1])

    # Compute y gradient (vertical)
    for i in range(1, h-1):
        for j in range(w):
            gy[i, j] = float(image[i+1, j]) - float(image[i-1, j])

    # Compute orientations (in degrees)
    orientations = np.zeros_like(image, dtype=float)
    for i in range(h):
        for j in range(w):
            orientations[i, j] = (np.arctan2(gy[i, j], gx[i, j]) * 180 / np.pi) % 180

    # Create orientation histogram (6 bins covering 0-180 degrees)
    n_bins = 6
    orient_hist = np.zeros(n_bins)
    bin_width = 180 / n_bins

    for i in range(n_bins):
        lower = i * bin_width
        upper = (i + 1) * bin_width
        count = 0
        for y in range(h):
            for x in range(w):
                if lower <= orientations[y, x] < upper:
                    # Weight by gradient magnitude for better results
                    magnitude = np.sqrt(gx[y, x]**2 + gy[y, x]**2)
                    if magnitude > 10:  # Threshold to reduce noise impact
                        count += 1
        orient_hist[i] = count

    # Normalize orientation histogram
    if np.sum(orient_hist) > 0:
        orient_hist = orient_hist / np.sum(orient_hist)

    # Add orientation histogram to features
    all_features.extend(orient_hist)

    # Compute gradient magnitude statistics
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    all_features.append(np.mean(gradient_magnitude))
    all_features.append(np.std(gradient_magnitude))

    # Process each region for GLCM (keep your existing GLCM code)
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

            # NEW: Add regional orientation histograms
            region_gx = gx[start_row:end_row, start_col:end_col]
            region_gy = gy[start_row:end_row, start_col:end_col]
            region_orientations = orientations[start_row:end_row, start_col:end_col]

            # Regional orientation histogram
            region_orient_hist = np.zeros(n_bins)
            for bin_idx in range(n_bins):
                lower = bin_idx * bin_width
                upper = (bin_idx + 1) * bin_width
                region_orient_hist[bin_idx] = np.sum((region_orientations >= lower) &
                                                     (region_orientations < upper))

            # Normalize
            if np.sum(region_orient_hist) > 0:
                region_orient_hist = region_orient_hist / np.sum(region_orient_hist)

            # Add to features
            all_features.extend(region_orient_hist)

    return np.array(all_features)

def extract_glcm_batch_from_scratch(images, batch_size=10):
    """
    Extract GLCM features from a batch of images using only NumPy

    Parameters:
    -----------
    images : ndarray
        Batch of grayscale images
    batch_size : int
        Size of mini-batches for processing

    Returns:
    --------
    features : ndarray
        GLCM features for all images
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
            features = extract_glcm_regional_features_from_scratch(img, num_regions=4, levels=16, distances=[1, 2, 3])
            batch_features.append(features)

        all_features.extend(batch_features)
        print(f"Processed {end}/{total} images")

    return np.array(all_features)

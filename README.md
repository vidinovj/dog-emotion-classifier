# Dog Emotion Classifier

A Streamlit application that classifies dog emotions into four categories (Sad, Angry, Happy, Relaxed) using GLCM, HOG, and MobileNetV2 features extractors and Support Vector Machines.

## Features

- Upload and process dog images
- Extract Gray Level Co-occurrence Matrix (GLCM) texture features
- Classify emotions using a pre-trained SVM model
- Visualize prediction confidence for each emotion category

## Directory

- File containing:  1. HOG-SVM app (no library)
                    2. HOG-SVM app (library)
- MobileNetV2 source code
- MobileNetV2 streamlit app
- GLCM-SVM source code
- GLCM-SVM streamlit app
- GLCM-SVM .pkl

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dog-emotion-classifier.git
cd dog-emotion-classifier

# Install dependencies
pip install streamlit

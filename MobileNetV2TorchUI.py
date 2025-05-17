import gradio as gr
import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load model and scaler
svm_model = joblib.load('dog_emotion_svm_model.joblib')
scaler = joblib.load('dog_emotion_scaler.joblib')

# Load emotion labels
with open('dog_emotion_features.pkl', 'rb') as f:
    data = joblib.load(f)
    emotion_labels = data['emotion_labels']

# Load feature extractor model (MobileNetV2 alternative from timm)
feature_extractor = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=0)
feature_extractor.eval()

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def predict_emotion(image: Image.Image):
    # Preprocess image
    input_tensor = preprocess(image).unsqueeze(0)

    # Extract features
    with torch.no_grad():
        features = feature_extractor(input_tensor).numpy()

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = svm_model.predict(features_scaled)[0]
    probabilities = svm_model.predict_proba(features_scaled)[0]

    # Format result
    result = f"Detected Emotion: **{emotion_labels[prediction].upper()}**\n\n"
    for label, prob in zip(emotion_labels, probabilities):
        result += f"{label.capitalize()}: {prob:.2%}\n"

    # Plot graph
    fig, ax = plt.subplots()
    sns.barplot(x=emotion_labels, y=probabilities, ax=ax)
    ax.set_title("Prediction Confidence")
    ax.set_ylim(0, 1)
    for i, prob in enumerate(probabilities):
        ax.text(i, prob + 0.02, f'{prob:.1%}', ha='center')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return result, buf

# Gradio interface
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Markdown(), gr.Image(type="file")],
    title="Dog Emotion Classifier 🐶",
    description="Upload a dog's image to classify its emotion (happy, sad, angry, relaxed)."
)

iface.launch()

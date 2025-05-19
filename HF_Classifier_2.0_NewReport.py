
import os
import torch
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import nltk

# Ensure NLTK stopwords are available
nltk.download('stopwords')
from nltk.corpus import stopwords

# Configuration
models_folder = "Models"
input_file = "new_report.txt"  # Path to the new report (.txt)
confidence_threshold = 0.6
max_length = 512
stride = 256

# Load and clean the report
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    return ' '.join(words)

with open(input_file, "r", encoding="utf-8") as file:
    raw_text = file.read()
cleaned_text = clean_text(raw_text)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Chunk text for BERT
def chunk_text(text, tokenizer, max_length=512, stride=256):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        if not chunk:
            continue
        chunk = tokenizer.build_inputs_with_special_tokens(chunk)
        chunks.append(chunk)
        if i + max_length >= len(tokens):
            break
    return chunks

# Predict using a model
def predict_with_model(model, tokenizer, text, threshold=0.6):
    chunks = chunk_text(text, tokenizer)
    aggregated_probs = None

    for chunk in chunks:
        inputs = tokenizer.prepare_for_model(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1)

        if aggregated_probs is None:
            aggregated_probs = probs
        else:
            aggregated_probs += probs

    avg_probs = aggregated_probs / len(chunks)
    confidence = torch.max(avg_probs).item()
    predicted_class = torch.argmax(avg_probs, dim=1).item()

    if confidence < threshold:
        return None, confidence, avg_probs.tolist()
    return predicted_class, confidence, avg_probs.tolist()

# Run predictions
predictions = []
for model_name in os.listdir(models_folder):
    model_path = os.path.join(models_folder, model_name)
    if not os.path.isdir(model_path):
        continue

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    predicted_class, confidence, probabilities = predict_with_model(model, tokenizer, cleaned_text, confidence_threshold)

    if predicted_class is not None:
        predictions.append({
            "model_name": model_name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        })

# Output results
print("=== Prediction Results ===")
for pred in predictions:
    print(f"Model: {pred['model_name']}")
    print(f"Predicted Class: {pred['predicted_class']}")
    print(f"Confidence: {pred['confidence']:.2f}")
    print()

import sys
import pandas as pd
import torch
import joblib
from PIL import Image

from transformers import AutoImageProcessor, ViTForImageClassification

from config import EXCEL_FILE

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model
# -----------------------------
print("Loading model...")

processor = AutoImageProcessor.from_pretrained("saved_model")

model = ViTForImageClassification.from_pretrained("saved_model")
model.to(DEVICE)
model.eval()

# -----------------------------
# Load label encoder
# -----------------------------
label_encoder = joblib.load("saved_model/label_encoder.pkl")

# -----------------------------
# Load Excel
# -----------------------------
df = pd.read_excel(EXCEL_FILE)

# -----------------------------
# Image path
# -----------------------------
if len(sys.argv) < 2:
    print("Usage:")
    print("python predict.py <image_path>")
    exit()

image_path = sys.argv[1]

# -----------------------------
# Open image
# -----------------------------
image = Image.open(image_path).convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

probs = torch.softmax(logits, dim=1)

confidence, pred = torch.max(probs, dim=1)

pred_class = label_encoder.inverse_transform([pred.item()])[0]

# -----------------------------
# Plant Information
# -----------------------------
plant = df[df["scientific_name"] == pred_class].iloc[0]

print("\n==============================")
print("Prediction Result")
print("==============================")

print(f"Scientific Name : {plant['scientific_name']}")
print(f"Local Name      : {plant['local_name']}")
print(f"Confidence      : {confidence.item()*100:.2f}%")

print("\nMedicinal Uses")
print(plant["medicinal_uses"])

print("\nParts Used")
print(plant["parts_used"])

print("\nOther Uses")
print(plant["Other Uses"])

print("\nCultivation Details")
print(plant["Cultivation Details"])
import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import EXCEL_FILE, DATASET_ROOT

print("Loading Excel...")

df = pd.read_excel(EXCEL_FILE)

old_root = "/content/drive/MyDrive/herb_dataset/Herbify-Dataset"

df["image_path"] = (
    df["image_path"]
    .str.replace(old_root, str(DATASET_ROOT), regex=False)
    .str.replace("/", os.sep)
)

# Keep only existing images
df = df[df["image_path"].apply(os.path.exists)].copy()

# Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["scientific_name"])

# Remove classes with fewer than 2 images (same as training)
counts = df["label"].value_counts()
valid = counts[counts >= 2].index
df = df[df["label"].isin(valid)].copy()

# Refit encoder after filtering
label_encoder = LabelEncoder()
label_encoder.fit(df["scientific_name"])

joblib.dump(label_encoder, "saved_model/label_encoder.pkl")

print("✅ label_encoder.pkl saved successfully!")
print("Number of classes:", len(label_encoder.classes_))
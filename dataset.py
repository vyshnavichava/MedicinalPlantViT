import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoImageProcessor

from config import (
    EXCEL_FILE,
    DATASET_ROOT,
    MODEL_NAME,
    RANDOM_SEED
)


class MedicinalPlantDataset(Dataset):
    def __init__(self, dataframe, processor, label_encoder):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")

        encoding = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = encoding["pixel_values"].squeeze(0)

        label = self.label_encoder.transform(
            [row["scientific_name"]]
        )[0]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_data():

    print("=" * 50)
    print("Loading Dataset...")
    print("=" * 50)

    # -------------------------
    # Load Excel
    # -------------------------

    df = pd.read_excel(EXCEL_FILE)

    print(f"Total rows in Excel : {len(df)}")

    # -------------------------
    # Convert Google Colab paths
    # -------------------------

    old_root = "/content/drive/MyDrive/herb_dataset/Herbify-Dataset"

    df["image_path"] = (
        df["image_path"]
        .str.replace(old_root, str(DATASET_ROOT), regex=False)
        .str.replace("/", os.sep)
    )

    # -------------------------
    # Keep only existing images
    # -------------------------

    df = df[df["image_path"].apply(os.path.exists)]

    print(f"Images found : {len(df)}")

    # -------------------------
    # Encode labels
    # -------------------------

    label_encoder = LabelEncoder()

    df["label"] = label_encoder.fit_transform(df["scientific_name"])

    print(f"Total classes before filtering : {len(label_encoder.classes_)}")

    # -------------------------
    # Remove classes having only one image
    # -------------------------

    class_counts = df["label"].value_counts()

    valid_labels = class_counts[class_counts >= 2].index

    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    # Re-encode labels after filtering
    label_encoder = LabelEncoder()

    df["label"] = label_encoder.fit_transform(df["scientific_name"])

    print(f"Remaining images : {len(df)}")
    print(f"Remaining classes : {len(label_encoder.classes_)}")

    # -------------------------
    # Train Validation Split
    # -------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    print(f"Training images : {len(train_df)}")
    print(f"Validation images : {len(val_df)}")

    # -------------------------
    # Image Processor
    # -------------------------

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    # -------------------------
    # Create datasets
    # -------------------------

    train_dataset = MedicinalPlantDataset(
        train_df,
        processor,
        label_encoder
    )

    val_dataset = MedicinalPlantDataset(
        val_df,
        processor,
        label_encoder
    )

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)

    return (
        train_dataset,
        val_dataset,
        label_encoder,
        processor
    )
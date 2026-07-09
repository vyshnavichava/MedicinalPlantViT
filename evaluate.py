import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from transformers import ViTForImageClassification

from dataset import load_data
from config import BATCH_SIZE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("Loading Dataset...")
print("=" * 50)

train_dataset, val_dataset, label_encoder, processor = load_data()

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("=" * 50)
print("Loading Trained Model...")
print("=" * 50)

model = ViTForImageClassification.from_pretrained(
    "saved_model"
)

model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []

print("Evaluating...\n")

with torch.no_grad():

    for batch in val_loader:

        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(pixel_values=pixel_values)

        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())



accuracy = accuracy_score(all_labels, all_preds)

precision = precision_score(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0,
)

print("=" * 50)
print("Evaluation Results")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

print("\nConfusion Matrix\n")

print(confusion_matrix(all_labels, all_preds))
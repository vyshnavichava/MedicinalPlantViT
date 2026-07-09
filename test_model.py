from dataset import load_data
from model import build_model

train_dataset, val_dataset, label_encoder, processor = load_data()

model = build_model(len(label_encoder.classes_))

print("\nModel Created Successfully")
print("-" * 40)
print("Classes :", model.config.num_labels)
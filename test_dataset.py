from dataset import load_data

train_dataset, val_dataset, label_encoder, processor = load_data()

print("\nDataset Summary")
print("---------------------------")

print("Train Samples :", len(train_dataset))
print("Validation Samples :", len(val_dataset))
print("Classes :", len(label_encoder.classes_))

sample = train_dataset[0]

print("\nSample Information")
print("---------------------------")

print("Pixel Values Shape :", sample["pixel_values"].shape)
print("Label :", sample["labels"])
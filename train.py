import os
import numpy as np
from transformers import TrainingArguments, Trainer

from dataset import load_data
from model import build_model

# Load dataset
train_dataset, val_dataset, label_encoder, processor = load_data()

# Build model
model = build_model(len(label_encoder.classes_))

# Compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Training arguments
training_args = TrainingArguments(
    output_dir="checkpoints",

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=3,

    weight_decay=0.01,

    load_best_model_at_end=True,

    remove_unused_columns=False,

    report_to="none",

    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

trainer.train()

os.makedirs("saved_model", exist_ok=True)

trainer.save_model("saved_model")
processor.save_pretrained("saved_model")

print("\nTraining completed successfully!")
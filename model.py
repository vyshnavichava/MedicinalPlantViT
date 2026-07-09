import torch
from transformers import ViTForImageClassification
from config import MODEL_NAME


def build_model(num_classes):
    """
    Build Vision Transformer model for medicinal plant classification.
    """

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    return model


if __name__ == "__main__":

    model = build_model(89)

    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel Summary")
    print("-" * 40)
    print("Total Parameters     :", f"{total_params:,}")
    print("Trainable Parameters :", f"{trainable_params:,}")
from pathlib import Path


# PROJECT PATHS


PROJECT_ROOT = Path(__file__).parent

DATASET_ROOT = PROJECT_ROOT / "Herbify-Dataset"

EXCEL_FILE = PROJECT_ROOT / "Copy of final_herbify_dataset.xlsx"

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# MODEL CONFIG


MODEL_NAME = "google/vit-base-patch16-224"

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 10

LEARNING_RATE = 2e-5

RANDOM_SEED = 42
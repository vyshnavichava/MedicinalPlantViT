# 🌿 MedicinalPlantViT

## Short Description
MedicinalPlantViT is a Streamlit application that identifies medicinal plants from leaf images using a Vision Transformer (ViT) model. It provides an interactive dashboard for plant recognition, prediction details, and medicinal information.

## Features
- Upload leaf images in JPG, JPEG, or PNG format
- Predict scientific name and common name
- Display confidence score and prediction time
- Show top 3 class predictions
- Expandable sections for:
  - Medicinal uses
  - Parts used
  - Other uses
  - Cultivation details
- Streamlit-based dashboard with responsive UI

## Tech Stack
- Python
- Streamlit
- PyTorch / Vision Transformer (ViT)
- scikit-learn
- NumPy
- pandas

## Dataset
- Dataset directory: `Herbify-Dataset/`
- Contains medicinal plant leaf images organized by species
- Used to train and validate the ViT classification model

## Model Architecture
- Vision Transformer (ViT) based image classification
- Trained on medicinal plant leaf images
- Predicts plant species and provides confidence values

## Project Structure
- `app.py` — Main Streamlit application and UI
- `config.py` — Configuration and constants
- `dataset.py` — Dataset loading and preprocessing utilities
- `model.py` — Model loading and inference functions
- `train.py` — Training script for model fine-tuning
- `evaluate.py` — Model evaluation and validation logic
- `predict.py` — Prediction helper functions
- `save_label_encoder.py` — Encoder saving utilities
- `requirements.txt` — Python project dependencies
- `README.md` — Project overview and usage instructions
- `saved_model/` — Saved model weights and tokenizer/config files
- `Herbify-Dataset/` — Raw dataset images

## Repository Layout
```
MedicinalPlantViT
│
├── assets/
│   ├── home.png
│   ├── upload.png
│   ├── prediction.png
│   └── details.png
```

## Installation
1. Clone the repository:

```bash
git clone https://github.com/vyshnavichava/MedicinalPlantViT.git
cd MedicinalPlantViT
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run
Start the Streamlit app locally:

```bash
streamlit run app.py
```

Open the URL shown in the terminal, usually `http://localhost:8501`.

## Screenshots
Add your app screenshots here if available:

![Home screen](assets/home.png)

![Upload screen](assets/upload.png)

![Prediction screen](assets/prediction.png)

![Details screen](assets/details.png)

## Results
- Model type: Vision Transformer (ViT)
- Dataset: Medicinal plant leaf images
- Example accuracy: 97.03% (reported during evaluation)
- Provides predictive labels with confidence scores and supplemental plant details

## Future Work
- Add support for more plant species and more training data
- Improve model accuracy with advanced fine-tuning
- Add image augmentation and preprocessing pipelines
- Deploy the app to a hosted platform like Streamlit Cloud or Heroku
- Add better documentation and user guidance in the UI

## Author
- Developed by Vyshnavichava
- GitHub: https://github.com/vyshnavichava

## Changelog (recent)
- Added Vision Transformer inference pipeline
- Improved prediction UI
- Integrated medicinal plant database
- Added confidence visualization
- Fixed image preprocessing
- Updated README
- Added deployment instructions

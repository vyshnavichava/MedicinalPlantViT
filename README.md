# Medicinal Plant Identification

A Streamlit app that identifies medicinal plants from leaf images using a Vision Transformer (ViT) model. The application provides a polished dashboard experience for plant recognition and medicinal information lookup.

## Key Features
- Upload leaf images in JPG, JPEG, or PNG format
- Predict scientific name and common name of the plant
- Show confidence score and prediction time
- Display top-3 predictions in a clean table
- Provide expandable sections for:
  - Medicinal uses
  - Parts used
  - Other uses
  - Cultivation details
- Professional green-themed dashboard layout for medical/plant applications

## Project Structure
- `app.py`: Main Streamlit application and UI layout
- `model.py`: Model loading and inference utilities
- `dataset.py`: Dataset loading and preprocessing helpers
- `train.py`: Training script for the ViT model
- `evaluate.py`: Evaluation and validation script
- `predict.py`: Prediction utilities
- `saved_model/`: Pretrained/fine-tuned model files
- `Herbify-Dataset/`: Original plant image dataset

## Installation
1. Clone the repository to your local machine.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the important files and directories are present:
- `app.py`
- `saved_model/`
- `Copy of final_herbify_dataset.xlsx`
- `requirements.txt`

## Usage
Run the app locally:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal (usually `http://localhost:8501`).

## App Output
After uploading an image, the app displays:
- A centered image preview with responsive sizing and rounded corners
- A Prediction Details card with:
  - Scientific Name
  - Common Name
  - Confidence
  - Prediction Time
  - Model
  - Confidence Level
- A Top 3 Predictions table aligned with the prediction section
- Expandable content blocks for medicinal uses, parts used, other uses, and cultivation details

### Screenshots
Add your app output screenshots here if available:

![App Output 1](outputs/output1.png)

![App Output 2](outputs/output2.png)

![App Output 3](outputs/output3.png)

![App Output 4](outputs/output4.png)

## Model Information
- Model: Vision Transformer (ViT)
- Dataset: 6,057 images covering 89 medicinal plant species
- Displayed accuracy: 97.03%

## Deployment
To deploy on Streamlit Community Cloud:
1. Push the repository to GitHub.
2. Go to https://share.streamlit.io/cloud-getting-started.
3. Create a new app and select:
   - Repository
   - Branch: `main`
   - File path: `app.py`
4. Deploy and open the published app link.

## Notes
- Keep `requirements.txt` up to date.
- If the model files are large, consider a deployment strategy that supports large data or Git LFS.
- If using a private repo, make sure Streamlit Cloud has access.

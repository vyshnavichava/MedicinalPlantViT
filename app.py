import streamlit as st
import torch
import pandas as pd
import joblib
import time

from PIL import Image
from transformers import (
    ViTForImageClassification,
    AutoImageProcessor
)


# Page Config

st.set_page_config(
    page_title="🌿 Medicinal Plant Identification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg,#eef9f0,#ffffff);
}

/* Main container */
.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
    max-width:1200px;
}

/* Title */
h1{
    text-align:center;
    color:#1B5E20;
    font-weight:700;
}

.page-header{
    display:flex;
    justify-content:center;
    align-items:center;
    min-height:30vh;
    text-align:center;
    margin:0.5rem 0 1rem;
    padding:0.5rem 1rem 1rem;
}

.page-header h2{
    font-size:2.3rem;
    margin-bottom:0.7rem;
}

.page-header .page-description{
    font-size:1.05rem;
    color:#4E6B4F;
    margin-top:0.5rem;
}

.footer-section{
    display:flex;
    justify-content:center;
    align-items:center;
    text-align:center;
    margin:3rem auto 2rem;
    padding:1.5rem 0;
    width:100%;
}

.footer-section > div {
    width:100%;
    display:flex;
    justify-content:center;
    align-items:center;
}

.footer-text{
    font-size:1rem;
    line-height:1.7;
    color:#2E7D32;
    margin:0 auto;
    text-align:center;
}

/* Cards */
.card{
    background:rgba(255,255,255,0.95);
    border:1px solid #dcefe0;
    border-radius:18px;
    padding:1rem 1.1rem;
    box-shadow:0 8px 24px rgba(38,85,46,0.08);
    margin-bottom:1rem;
}

.upload-card{
    padding:1rem 1.2rem;
}

.card-title{
    font-size:1.1rem;
    font-weight:700;
    color:#2E7D32;
    margin-bottom:0.8rem;
}

.results-shell{
    max-width:1100px;
    margin:0 auto 1rem;
}

.result-card{
    height:100%;
}

.image-frame{
    display:flex;
    justify-content:center;
    align-items:center;
    background:linear-gradient(135deg,#f7fff8,#ffffff);
    border:1px solid #e2efe3;
    border-radius:16px;
    padding:0.7rem;
}

.image-frame img{
    border-radius:14px;
    max-height:320px;
    object-fit:cover;
    width:100%;
}

.image-caption{
    text-align:center;
    color:#5f6f5f;
    margin-top:0.45rem;
    font-size:0.95rem;
}

.detail-card{
    background:linear-gradient(135deg,#f7fff8,#ffffff);
    border:1px solid #dcefe0;
    border-radius:16px;
    padding:1rem 1.1rem;
    height:100%;
}

.detail-row{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:0.8rem;
    padding:0.45rem 0;
    border-bottom:1px solid #edf5ee;
    font-size:0.96rem;
}

.detail-row:last-child{
    border-bottom:none;
}

.detail-label{
    color:#4E6B4F;
    font-weight:600;
}

.detail-value{
    color:#1B5E20;
    font-weight:700;
    text-align:right;
}

.badge{
    display:inline-block;
    padding:0.45rem 0.8rem;
    border-radius:999px;
    margin-top:0.75rem;
    font-weight:700;
    font-size:0.9rem;
}

.badge-high{
    background:#E8F5E9;
    color:#2E7D32;
}

.badge-medium{
    background:#FFF8E1;
    color:#C47A00;
}

.badge-low{
    background:#FFEBEE;
    color:#C62828;
}

.top3-table{
    width:100%;
    border-collapse:collapse;
    margin-top:0.2rem;
}

.top3-table th,
.top3-table td{
    padding:0.7rem 0.75rem;
    border-bottom:1px solid #edf5ee;
    text-align:left;
}

.top3-table th{
    color:#2E7D32;
    font-weight:700;
}

.top3-table tr.highlight-row{
    background:#F3FFF4;
    font-weight:700;
    color:#1B5E20;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#E8F5E9;
}

/* Metric cards */
[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0 3px 10px rgba(0,0,0,.08);
    border:1px solid #dfe8df;
}

/* Expanders */
.streamlit-expanderHeader{
    font-size:18px;
    font-weight:bold;
}

/* Buttons */
.stButton>button{
    background:#2E7D32;
    color:white;
    border-radius:10px;
    border:none;
    height:45px;
    width:100%;
}

.stButton>button:hover{
    background:#1B5E20;
}

/* File uploader */
[data-testid="stFileUploader"]{
    border:2px dashed #43A047;
    border-radius:15px;
    padding:15px;
    background:#F8FFF8;
}

/* Images */
img{
    border-radius:15px;
}

/* Success box */
[data-testid="stAlert"]{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class='page-header'>
    <div>
        <h2>🌿 Medicinal Plant Identification System</h2>
        <div>Vision Transformer (ViT)</div>
        <div class='page-description'>Upload a medicinal leaf image to identify the plant species and explore its medicinal information.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Paths

MODEL_PATH = "saved_model"
LABEL_ENCODER = "saved_model/label_encoder.pkl"
EXCEL_FILE = "Copy of final_herbify_dataset.xlsx"

# Load Model

@st.cache_resource
def load_model():

    label_encoder = joblib.load(LABEL_ENCODER)

    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

    model = ViTForImageClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(label_encoder.classes_),
        ignore_mismatched_sizes=True
    )

    model.eval()

    return model, processor, label_encoder


model, processor, label_encoder = load_model()

# Load Excel Dataset

@st.cache_data
def load_dataframe():
    return pd.read_excel(EXCEL_FILE)


df = load_dataframe()
with st.sidebar:
    st.title("🌿 Navigation")
    st.markdown("---")
    st.success("Model Ready")
    st.metric("Accuracy", "97.03%")
    st.metric("Dataset", "89 species")
    st.metric("Images", "6057")
    st.markdown("---")
    st.subheader("📌 Quick Start")
    st.write("""
    1️⃣ Upload Leaf Image

    2️⃣ AI analyzes image

    3️⃣ Plant is identified

    4️⃣ View medicinal information
    """)

with st.container():
    st.markdown("<div class='card upload-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📤 Upload Leaf Image</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown("<div class='small-text'>Supported formats: JPG, JPEG, PNG</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    start = time.time()
    image = Image.open(uploaded_file).convert("RGB")

    top_probs, top_indices = None, None
    confidence, predicted_class, row, probabilities = None, None, None, None
    score = None

    inputs = processor(images=image, return_tensors="pt")
    with st.spinner("🔍 AI is analyzing the medicinal leaf..."):
        with torch.no_grad():
            outputs = model(**inputs)
        end = time.time()

    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    score = confidence.item() * 100
    scientific_name = label_encoder.inverse_transform([predicted_class.item()])[0]
    row = df[df["scientific_name"].str.lower() == scientific_name.lower()].iloc[0]
    top_probs, top_indices = torch.topk(probabilities, 3)

    with st.container():
        st.markdown("<div class='results-shell'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1], gap="small")

        with col1:
            st.markdown("<div class='card result-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📷 Uploaded Image</div>", unsafe_allow_html=True)
            st.markdown("<div class='image-frame'>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='image-caption'>Uploaded Leaf</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card result-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📊 Prediction Details</div>", unsafe_allow_html=True)

            if score > 80:
                badge_html = "<div class='badge badge-high'>High Confidence</div>"
            elif score >= 50:
                badge_html = "<div class='badge badge-medium'>Medium Confidence</div>"
            else:
                badge_html = "<div class='badge badge-low'>Low Confidence</div>"

            detail_rows = [
                ("Scientific Name", scientific_name.title()),
                ("Common Name", row["local_name"]),
                ("Confidence", f"{score:.2f}%"),
                ("Prediction Time", f"{end - start:.2f} sec"),
                ("Model", "Vision Transformer (ViT)"),
                ("Confidence Level", badge_html)
            ]

            detail_html = "<div class='detail-card'>"
            for label, value in detail_rows:
                if label == "Confidence Level":
                    detail_html += f"<div class='detail-row'><div class='detail-label'>{label}</div><div class='detail-value'>{value}</div></div>"
                else:
                    detail_html += f"<div class='detail-row'><div class='detail-label'>{label}</div><div class='detail-value'>{value}</div></div>"
            detail_html += "</div>"
            st.markdown(detail_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🏆 Top 3 Predictions</div>", unsafe_allow_html=True)
        top_predictions = []
        for i in range(3):
            plant = label_encoder.inverse_transform([top_indices[0][i].item()])[0]
            prob = round(top_probs[0][i].item() * 100, 2)
            top_predictions.append({"Rank": i + 1, "Plant": plant.title(), "Confidence": f"{prob:.2f}%"})

        table_rows = ""
        for idx, pred in enumerate(top_predictions):
            row_class = "highlight-row" if idx == 0 else ""
            table_rows += f"<tr class='{row_class}'><td>{pred['Rank']}</td><td>{pred['Plant']}</td><td>{pred['Confidence']}</td></tr>"

        st.markdown(f"""
        <table class='top3-table'>
            <thead>
                <tr><th>Rank</th><th>Plant</th><th>Confidence</th></tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        with st.expander("📌 Medicinal Uses", expanded=False):
            st.write(row["medicinal_uses"])
    with st.container():
        with st.expander("🌿 Parts Used", expanded=False):
            st.write(row["parts_used"])
    with st.container():
        with st.expander("🍃 Other Uses", expanded=False):
            st.write(row["Other Uses"])
    with st.container():
        with st.expander("🌱 Cultivation Details", expanded=False):
            st.write(row["Cultivation Details"])

with st.container():
    st.markdown("<div class='footer-section'>", unsafe_allow_html=True)
    st.markdown("<div class='footer-text'>Developed by<br>Vyshnavi Chava<br>SRM University AP<br>Vision Transformer (ViT)<br>Version 1.0</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


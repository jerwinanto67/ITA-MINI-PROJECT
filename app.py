"""
CropSense - Streamlit web app.

Run:  streamlit run app.py
"""
import os

import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "crop_model.pkl")
DATA_PATH = os.path.join(BASE, "data", "Crop_recommendation.csv")
OUT_DIR = os.path.join(BASE, "outputs")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

CROP_TIPS = {
    "rice": "Needs standing water and high humidity. Best in clayey, water-retaining soil.",
    "maize": "Grows well in well-drained loamy soil with moderate rainfall.",
    "chickpea": "A rabi pulse. Prefers cool, dry weather and light soils.",
    "kidneybeans": "Needs well-drained soil and moderate temperature; avoid waterlogging.",
    "pigeonpeas": "Drought tolerant pulse; suits semi-arid regions.",
    "mothbeans": "Very drought hardy; suits sandy soils with low rainfall.",
    "mungbean": "Short-duration pulse; needs warm weather and light irrigation.",
    "blackgram": "Grows in warm humid climate; good for rice fallows.",
    "lentil": "Cool-season crop; needs little water and well-drained soil.",
    "pomegranate": "Suits semi-arid climate; tolerant to drought and salinity.",
    "banana": "Needs rich, well-drained soil, high humidity and regular watering.",
    "mango": "Tropical fruit; needs a dry spell before flowering.",
    "grapes": "Needs warm dry climate and well-drained soil; requires pruning.",
    "watermelon": "Warm-season crop; sandy loam soil and plenty of sunlight.",
    "muskmelon": "Warm, dry weather and sandy loam soil give sweeter fruit.",
    "apple": "Needs cool climate and chilling hours; grows in hilly regions.",
    "orange": "Subtropical fruit; well-drained soil and regular irrigation.",
    "papaya": "Fast-growing tropical fruit; avoid waterlogging.",
    "coconut": "Coastal crop; needs high humidity and sandy, well-drained soil.",
    "cotton": "Needs black soil, warm climate and moderate rainfall.",
    "jute": "Needs hot humid climate, heavy rainfall and alluvial soil.",
    "coffee": "Grows on hill slopes with shade, high rainfall and rich soil.",
}

st.set_page_config(page_title="CropSense", page_icon="🌱", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


page = st.sidebar.radio("Navigate", ["Recommend a crop", "Dataset insights", "About"])

if page == "Recommend a crop":
    st.title("🌱 CropSense")
    st.write("Enter your soil and weather details to get the most suitable crop.")

    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)
        n = c1.number_input("Nitrogen (N)", 0, 200, 90)
        p = c2.number_input("Phosphorus (P)", 0, 200, 42)
        k = c3.number_input("Potassium (K)", 0, 250, 43)
        c4, c5 = st.columns(2)
        temp = c4.number_input("Temperature (°C)", 0.0, 50.0, 21.0, 0.1)
        hum = c5.number_input("Humidity (%)", 0.0, 100.0, 82.0, 0.1)
        c6, c7 = st.columns(2)
        ph = c6.number_input("Soil pH", 0.0, 14.0, 6.5, 0.1)
        rain = c7.number_input("Rainfall (mm)", 0.0, 400.0, 203.0, 0.1)
        submitted = st.form_submit_button("Recommend crop")

    if submitted:
        model = load_model()
        if model is None:
            st.error("Model file not found! Please run `python train_model.py` first to train and save the model.")
        else:
            x = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=FEATURES)
            crop = model.predict(x)[0]
            proba = model.predict_proba(x)[0]
            top = sorted(zip(model.classes_, proba), key=lambda t: -t[1])[:3]

            st.success(f"Recommended crop: **{crop.title()}**")
            st.info(CROP_TIPS.get(crop, "Optimal water and care recommended for high yield."))
            st.subheader("Top 3 suggestions")
            st.dataframe(
                pd.DataFrame([(c.title(), f"{pr*100:.1f}%") for c, pr in top], columns=["Crop", "Confidence"]),
                hide_index=True,
                use_container_width=True,
            )

elif page == "Dataset insights":
    st.title("📊 Dataset insights")
    df = load_data()
    if df is not None:
        st.write(f"{len(df)} records, {df['label'].nunique()} crops, 7 input features.")
        st.dataframe(df.head(10), use_container_width=True)
        crop = st.selectbox("Average conditions for a crop", sorted(df["label"].unique()))
        st.bar_chart(df[df["label"] == crop][FEATURES].mean())
        for img, cap in [
            ("model_comparison.png", "Model accuracy comparison"),
            ("feature_importance.png", "Feature importance"),
            ("confusion_matrix.png", "Confusion matrix of best model"),
            ("feature_distribution.png", "Distribution of input features"),
            ("correlation_heatmap.png", "Feature correlation heatmap"),
        ]:
            path = os.path.join(OUT_DIR, img)
            if os.path.exists(path):
                st.image(path, caption=cap)
    else:
        st.error("Dataset not found! Ensure `data/Crop_recommendation.csv` exists.")

else:
    st.title("About CropSense")
    st.write(
        "CropSense is a machine learning based crop recommendation system. "
        "It uses soil nutrients (N, P, K), soil pH and climate data (temperature, humidity, rainfall) "
        "to suggest the most suitable crop. The model is trained on the Kaggle Crop Recommendation dataset."
    )

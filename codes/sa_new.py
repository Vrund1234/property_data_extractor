import streamlit as st
import numpy as np
import cv2
import json
import tempfile
import os
from google import genai
from pydantic import BaseModel
from PIL import Image
import pandas as pd
from deep_translator import GoogleTranslator


# =============== CONFIG ===============
GEMINI_API_KEY = "AIzaSyBdyEnkr0vnwqva4nULt1rY0_9QCfm5gDc"
client = genai.Client(api_key=GEMINI_API_KEY)

translator = GoogleTranslator(source='auto', target='en')


# =============== RESPONSE SCHEMA ===============
class ClassifiedAd(BaseModel):
    Type: str | None
    Vaar: str | None
    SqFeet: str | None
    Address: str | None
    Contact_Number: str | None
    Property_For: str | None


# =============== SEGMENT FUNCTION ===============
def segment_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    column_sums = np.sum(thresh, axis=0)
    threshold = np.max(column_sums) * 0.1
    column_boundaries = np.where(column_sums > threshold)[0]

    column_positions = []
    start = column_boundaries[0]
    for i in range(1, len(column_boundaries)):
        if column_boundaries[i] - column_boundaries[i - 1] > 10:
            column_positions.append((start, column_boundaries[i - 1]))
            start = column_boundaries[i]
    column_positions.append((start, column_boundaries[-1]))

    image_pil = Image.open(image_path)
    column_paths = []
    for i, (x1, x2) in enumerate(column_positions):
        col_img = image_pil.crop((x1, 0, x2, image_pil.height))
        if col_img.mode == "RGBA":
            col_img = col_img.convert("RGB")

        col_path = f"column_{i + 1}.jpg"
        col_img.save(col_path, "JPEG")
        column_paths.append(col_path)

    return column_paths


# =============== EXTRACT FUNCTION (ONE CALL) ===============
def extract_ads(column_image_paths):
    structured_ads = []

    prompt = """
    Extract structured property details from the classified ads in these images.

    Example output:
    {
        "ads": [
            {"Type": "2 BHK Flat", "Vaar": "1500", "SqFeet": "400", "Address": "Manekbaug Ambawadi", "Contact_Number": "9909935387", "Property_For": "For Sale"}
        ]
    }

    Do not change the extracted text; return it exactly.
    """

    images = [Image.open(p) for p in column_image_paths]

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt] + images,
        config={"response_mime_type": "application/json"},
    )

    try:
        response_text = gemini_response.candidates[0].content.parts[0].text
        response_data = json.loads(response_text)
        structured_ads = response_data.get("ads", [])
    except Exception as e:
        st.error(f"Error extracting ads: {e}")

    return structured_ads


# =============== TRANSLATE FUNCTION ===============
def translate_ads(ads):
    translated_ads = []
    for ad in ads:
        translated_ad = {}
        for key, value in ad.items():
            if value:
                translated_value = translator.translate(value)
                translated_ad[key] = translated_value
            else:
                translated_ad[key] = None
        translated_ads.append(translated_ad)
    return translated_ads


# =============== STREAMLIT APP ===============
st.set_page_config(page_title="Classified Ads OCR Extractor")

st.title("📰 Classified Ads OCR Extractor")
st.write(
    "Upload a scanned newspaper page — this app will detect columns, run OCR using Gemini 2.5, and extract & translate property ads to English."
)

uploaded_file = st.file_uploader(
    "📤 Upload your newspaper image (JPG, PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_image_path = tmp_file.name

    st.image(temp_image_path, caption=f"Uploaded Scan", use_container_width=True)

    if st.button("🔍 Extract & Translate Ads"):
        with st.spinner("Segmenting columns, extracting ads, and translating to English..."):
            column_images = segment_image(temp_image_path)
            ads = extract_ads(column_images)

            if ads:
                translated_ads = translate_ads(ads)
                df = pd.DataFrame(translated_ads)

                st.session_state["ads_df"] = df

                st.success(f"✅ Extracted & Translated {len(translated_ads)} ads!")

    if "ads_df" in st.session_state:
        st.dataframe(st.session_state["ads_df"])

        csv = st.session_state["ads_df"].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name="property_data.csv",
            mime="text/csv",
        )
    else:
        st.info("👆 Click 'Extract & Translate Ads' to see results.")

else:
    st.info("📌 Please upload an image to start.")

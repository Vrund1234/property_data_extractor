import streamlit as st
import numpy as np
import cv2
import json
import tempfile
import os
import google.generativeai as genai
from pydantic import BaseModel
from PIL import Image
import pandas as pd
# from translate import Translator
import re

# =============== CONFIG ===============
GEMINI_API_KEY = "AIzaSyAnPBedVewH06WOgcc_ufnAZIU81XjMTo8"
genai.configure(api_key=GEMINI_API_KEY)

# translator = Translator(to_lang="en")

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

# =============== EXTRACT FUNCTION (SAFE PARSE) ===============
def extract_ads(column_image_paths):
    structured_ads = []

    prompt = """
You are an expert at extracting structured information from images. Carefully read the property ads from the images and return only valid JSON in the following format (no explanations):

{
  "ads": [
    {
      "Type": "",
      "Vaar": "",
      "SqFeet": "",
      "Address": "",
      "Contact_Number": "",
      "Property_For": ""
    },
    ...
  ]
}

⚠️ Do not write anything except the JSON. No explanations. No introductions.
    """

    images = [Image.open(p) for p in column_image_paths]

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')  # ✅ switched to 2.5
        gemini_response = model.generate_content(
            [prompt] + images,
            generation_config={"response_mime_type": "application/json"}
        )

        response_text = gemini_response.text

        # Clean JSON response
        json_text = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_text:
            response_data = json.loads(json_text.group())
            structured_ads = response_data.get("ads", [])
        else:
            st.error("❌ Could not find valid JSON in response.")

    except json.JSONDecodeError as e:
        st.error(f"❌ JSON Parsing error: {e}")
        st.text_area("Response Text", response_text, height=300)

    except Exception as e:
        st.error(f"❌ Error extracting ads: {e}")

    return structured_ads


# =============== TRANSLATE FUNCTION ===============
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='en')

def translate_ads(ads):
    translated_ads = []
    for ad in ads:
        translated_ad = {}
        for key, value in ad.items():
            if value:
                try:
                    translated_value = translator.translate(value)
                except Exception:
                    translated_value = value
                translated_ad[key] = translated_value
            else:
                translated_ad[key] = None
        translated_ads.append(translated_ad)
    return translated_ads


# =============== STREAMLIT APP ===============
st.set_page_config(page_title="Property Ads Extractor")

st.title("📰 Property Ads Extractor")

uploaded_file = st.file_uploader("📤 Upload your newspaper image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_image_path = tmp_file.name

    st.image(temp_image_path, caption="Uploaded Scan", use_container_width=True)

    if st.button("🔍 Extract & Translate Ads"):
        with st.spinner("Segmenting columns, extracting ads, and translating to English..."):
            column_images = segment_image(temp_image_path)
            ads = extract_ads(column_images)

            if ads:
                translated_ads = translate_ads(ads)
                df = pd.DataFrame(translated_ads)

                st.session_state["ads_df"] = df

                st.success(f"✅ Extracted & Translated {len(translated_ads)} ads!")
            else:
                st.warning("⚠️ No ads extracted. Please try with another image or check the image quality.")

    if "ads_df" in st.session_state:
        st.dataframe(st.session_state["ads_df"])

        csv = st.session_state["ads_df"].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, file_name="property_data.csv", mime="text/csv")
    else:
        st.info("👆 Click 'Extract & Translate Ads' to see results.")

else:
    st.info("📌 Please upload an image to start.")

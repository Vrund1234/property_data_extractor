import streamlit as st
import numpy as np
import cv2
import json
import tempfile
import io
from PIL import Image
import pandas as pd
import re
import pytesseract
import concurrent.futures
import google.generativeai as genai
from deep_translator import GoogleTranslator
from dateutil import parser

# =============== CONFIG ===============
GEMINI_API_KEY = "AIzaSyAnPBedVewH06WOgcc_ufnAZIU81XjMTo8"
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_translator():
    return GoogleTranslator(source='auto', target='en')

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel('gemini-2.5-flash')

translator = load_translator()
model = load_gemini_model()

# =============== Newspaper Detection + Date Extraction ===============
def detect_newspaper_name_and_date(image_path):
    try:
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        top_section = img[0:int(height * 0.50), :]

        upscale = cv2.resize(top_section, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscale, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        text_eng = pytesseract.image_to_string(enhanced, lang='eng').lower()
        text_guj = pytesseract.image_to_string(enhanced, lang='guj').lower()
        combined_text = text_eng + " " + text_guj

        known_newspapers = [
            "Gujarat Samachar", "Sandesh", "Nav Gujarat Samay",
            "Times of India", "Divya Bhaskar"
        ]

        detected_name = "Unknown Newspaper"
        for paper in known_newspapers:
            if paper.lower() in combined_text:
                detected_name = paper
                break

        date_pattern = r'\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\b'
        match = re.search(date_pattern, combined_text)
        detected_date = match.group(1) if match else "Unknown Date"

        if detected_date != "Unknown Date":
            try:
                parsed_date = parser.parse(detected_date, dayfirst=True)
                detected_date = parsed_date.strftime("%Y-%m-%d")
            except:
                pass

        st.write("📰 Detected Text (Partial):", combined_text[:500])
        st.write(f"✅ Newspaper: {detected_name} | 🗓️ Date: {detected_date}")

        return detected_name, detected_date

    except Exception as e:
        st.error(f"❌ Newspaper detection/date extraction error: {e}")
        return "Unknown Newspaper", "Unknown Date"

# =============== Segment Function ===============
def segment_image(image_path, resize_width=1200):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    if w > resize_width:
        image = cv2.resize(image, (resize_width, int(h * resize_width / w)), interpolation=cv2.INTER_AREA)

    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_sums = np.sum(thresh, axis=0)
    threshold = np.max(col_sums) * 0.1
    boundaries = np.where(col_sums > threshold)[0]

    if boundaries.size == 0:
        return []

    positions = []
    start = boundaries[0]
    for i in range(1, len(boundaries)):
        if boundaries[i] - boundaries[i - 1] > 10:
            positions.append((start, boundaries[i - 1]))
            start = boundaries[i]
    positions.append((start, boundaries[-1]))

    image_pil = Image.fromarray(image).convert("RGB")
    columns = []
    for x1, x2 in positions:
        buf = io.BytesIO()
        image_pil.crop((x1, 0, x2, image_pil.height)).save(buf, format="JPEG")
        buf.seek(0)
        columns.append(buf)

    return columns

# =============== Extract Function ===============
def extract_ads(column_buffers, newspaper_name, date):
    prompt = f"""
You are an expert at extracting property ads. Return only this JSON:

{{
  "ads": [
    {{
      "Type": "",
      "Property": "",
      "UOM": "",
      "Address": "",
      "Contact_Number": "",
      "Property_For": "",
      "Price": "",
      "Newspaper_Name": "{newspaper_name}",
      "Date": "{date}"
    }}
  ]
}}
No explanation. Just JSON.
"""

    images = [Image.open(buf) for buf in column_buffers]

    try:
        response = model.generate_content([prompt] + images, generation_config={"response_mime_type": "application/json"})
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            ads_data = json.loads(match.group())
            return ads_data.get("ads", [])
    except Exception as e:
        st.error(f"❌ Gemini extraction failed: {e}")
    return []

# =============== Translate & Clean Function ===============
def normalize_uom(uom_text):
    if not uom_text:
        return ""
    uom_text = uom_text.lower()
    if any(u in uom_text for u in ["sq ft", "sqft", "sq. ft", "square feet"]):
        return "SqFt"
    if any(u in uom_text for u in ["sq yard", "sq. yd", "square yard"]):
        return "SqYard"
    if "bigha" in uom_text:
        return "Bigha"
    if "acre" in uom_text:
        return "Acre"
    if "hectare" in uom_text:
        return "Hectare"
    return uom_text.title()

def translate_and_clean_ads(ads):
    junk_keywords = {"drop", "engine", "ad", "classified", "advertisement"}
    translated = []

    for ad in ads:
        cleaned_ad = {}
        for key, value in ad.items():
            if value and key not in ["Newspaper_Name", "Date"]:
                try:
                    value = translator.translate(value)
                except:
                    pass
                cleaned_ad[key] = value
            else:
                cleaned_ad[key] = value

        # Format Date
        if cleaned_ad["Date"] and cleaned_ad["Date"] != "Unknown Date":
            try:
                parsed_date = parser.parse(cleaned_ad["Date"], dayfirst=True)
                cleaned_ad["Date"] = parsed_date.strftime("%Y-%m-%d")
            except:
                pass

        # Address clean
        addr = cleaned_ad.get("Address", "").strip()
        addr = re.sub(r'[^A-Za-z0-9 ,.-]', '', addr)
        cleaned_ad["Address"] = addr if len(addr) >= 5 else ""

        # Price clean
        price = cleaned_ad.get("Price", "")
        price_numbers = re.findall(r'\d+', price)
        price_value = int(price_numbers[0]) if price_numbers else None
        unit = None
        price_lower = price.lower()

        if "lakh" in price_lower or "lac" in price_lower:
            unit = "Lakh"
        elif "crore" in price_lower:
            unit = "Crore"

        if not unit and price_value:
            if 1_00_000 <= price_value < 1_00_00_000:
                unit = "Lakh"
                price_value = price_value / 1_00_000
            elif price_value >= 1_00_00_000:
                unit = "Crore"
                price_value = price_value / 1_00_00_000

        if price_value and unit:
            cleaned_ad["Price"] = f"{price_value:.2f} {unit}"
        elif price_value:
            cleaned_ad["Price"] = str(price_value)

        # UOM normalization
        cleaned_ad["UOM"] = normalize_uom(cleaned_ad.get("UOM", ""))
        cleaned_ad["Property"] = re.sub(r'[^\d.]', '', cleaned_ad.get("Property", ""))

        # Skip junk ads
        if any(jk in cleaned_ad.get("Type", "").lower() for jk in junk_keywords):
            continue

        translated.append(cleaned_ad)

    return translated

# =============== Generate IDs ===============
def generate_ids(df):
    newspaper_map = {
        "gujarat samachar": "GUSM",
        "sandesh": "SNDE",
        "times of india": "TOIN",
        "nav gujarat samay": "NVGJ",
        "divya bhaskar": "DVBH"
    }

    ids = []
    for idx, row in df.iterrows():
        date_str = row["Date"].replace("-", "")
        paper = row["Newspaper_Name"].lower()
        code = "UNK"
        for name, abbr in newspaper_map.items():
            if name in paper:
                code = abbr
                break
        ids.append(f"{date_str}/{code}/{str(idx+1).zfill(4)}")

    df["ID"] = ids
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("ID")))
    return df[cols]

# =============== Full Pipeline ===============
def process_image(image_path):
    newspaper_name, date = detect_newspaper_name_and_date(image_path)
    columns = segment_image(image_path)
    if not columns:
        return []

    ads = []
    for col in columns:
        ads += extract_ads([col], newspaper_name, date)

    if not ads:
        return []

    return translate_and_clean_ads(ads)

# =============== Streamlit App ===============
st.set_page_config(page_title="📰 Property Ads Extractor")
st.title("📰 Property Ads Extractor")

uploaded_files = st.file_uploader("📤 Upload Newspaper Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    temp_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            temp_paths.append(tmp.name)
            st.image(tmp.name, caption=uploaded_file.name, use_container_width=True)

    if st.button("🔍 Extract & Translate Ads"):
        with st.spinner("Processing all images..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_image, temp_paths))

        all_ads = [ad for result in all_results for ad in result]

        if all_ads:
            df = pd.DataFrame(all_ads)
            df.drop_duplicates(inplace=True)
            df = generate_ids(df)
            st.session_state["ads_df"] = df
            st.success(f"✅ Extracted ads from {len(uploaded_files)} image(s)!")
        else:
            st.warning("⚠️ No ads found.")

    if "ads_df" in st.session_state:
        st.dataframe(st.session_state["ads_df"], use_container_width=True)
        csv = st.session_state["ads_df"].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, file_name="property_data.csv", mime="text/csv")

else:
    st.info("📌 Upload one or more newspaper images to begin.")

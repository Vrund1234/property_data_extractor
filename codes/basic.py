import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import pytesseract
import pandas as pd
import io

st.set_page_config(page_title="📰 Newspaper Image Scraper", layout="wide")
st.title("📰 Newspaper Image Scraper")

# Session state to store results
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []

# 1️⃣ Upload image
st.sidebar.header("📤 Upload Newspaper Image")
uploaded_file = st.sidebar.file_uploader("Upload a scanned newspaper image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # 2️⃣ Draw selection
    st.subheader("✏️ Select Area to Extract Text")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#e00",
        background_image=image,
        update_streamlit=True,
        height=image.height if image.height < 800 else 800,
        width=image.width if image.width < 1200 else 1200,
        drawing_mode="rect",
        key="canvas",
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            for obj in objects:
                left = int(obj["left"])
                top = int(obj["top"])
                width = int(obj["width"])
                height = int(obj["height"])

                # 3️⃣ Crop selected area
                cropped_img = image.crop((left, top, left + width, top + height))
                st.image(cropped_img, caption="🗂️ Selected Region")

                # 4️⃣ Run OCR
                extracted_text = pytesseract.image_to_string(cropped_img, lang="eng")

                st.write("🔍 **Extracted Text:**")
                edited_text = st.text_area("You can edit the extracted text here:", value=extracted_text, height=200)

                label = st.selectbox("Label this region as:", ["Headline", "Article", "Advertisement", "Other"])

                if st.button("✅ Save Extracted Text", key=f"save_{left}_{top}"):
                    st.session_state.extracted_data.append({
                        "Label": label,
                        "Extracted Text": edited_text
                    })
                    st.success("✅ Saved!")

# 5️⃣ Display extracted data
st.subheader("📑 Extracted Data")
if st.session_state.extracted_data:
    df = pd.DataFrame(st.session_state.extracted_data)
    st.dataframe(df)

    # Download as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download as CSV",
        data=csv,
        file_name="extracted_newspaper_data.csv",
        mime="text/csv",
    )

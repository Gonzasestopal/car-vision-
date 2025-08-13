import io

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Kopiloto Vision", page_icon="🚗", layout="centered")

st.title("🚗 Kopiloto Vision")
st.write("Upload car photos to analyze condition and estimate value.")

# File uploader (accepts multiple images)
uploaded_files = st.file_uploader(
    "Upload one or more car images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("Uploaded Images")
    for file in uploaded_files:
        # Open image
        image = Image.open(file)

        # Show image
        st.image(image, caption=file.name, use_column_width=True)

        # Placeholder for AI analysis
        st.info("🧠 AI analysis will appear here...")
        # Example: damage_results = detect_damage(image)
        # st.write(damage_results)

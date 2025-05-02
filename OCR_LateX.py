import streamlit as st
import base64
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import os
import requests
from pathlib import Path


# Load API key from .env file
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=api_key)
def call_groq_api(image_base64: str, prompt: str):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "Error: Missing GROQ_API_KEY environment variable"

    # Use the Meta Llama 4 Scout model (it supports images) rather than the 3.2 previews
    model = "meta-llama/llama-4-scout-17b-16e-instruct"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}" }}
                ]
            }
        ],
        # optional: enforce JSON-mode if you want structured output
        # "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 1000
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"], None

    except requests.exceptions.HTTPError as http_err:
        # Try to parse and show the API’s own error message
        try:
            err = http_err.response.json()
        except ValueError:
            err = http_err.response.text
        return None, f"Groq API returned {http_err.response.status_code}: {err}"

    except Exception as e:
        return None, f"Unexpected error calling Groq API: {e}"


#-------------------------------------
#          Main Content
#-------------------------------------
def extract_latex_from_image(uploaded_file):
    with st.spinner("Processing image..."):
        try:
            # Encode image to Base64
            buffered = BytesIO()
            image = Image.open(uploaded_file)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        "Extract the mathematical equation in the provided image as LaTeX code.",
                        "Follow these strict guidelines:",
                        "- Output only the LaTeX code without additional text.",
                        "- Do not simplify equations.",
                        "- Do not add documentclass, packages, or begindocument.",
                        "- Do not include dollar signs ($) around the LaTeX code.",
                        "- Do not explain symbols in the equation.",
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
            return response.choices[0].message.content  # Fix: Access content correctly
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None

#-------------------------------------
#          Streamlit UI
#-------------------------------------
# Streamlit page configuration
st.set_page_config(
    page_title="LaTeX OCR with GPT-4o",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Interface
st.title("👁️ LaTeX OCR with GPT-4o")
st.markdown('<p style="margin-top: -20px;">Extract LaTeX code from images using GPT-4o!</p>', unsafe_allow_html=True)


#-------------------------------------
#         Two Column Boxed Layout
#-------------------------------------
col1, col2 = st.columns(2, border=True)

# Upload Image
with col1:
    st.header("📤 Upload Image")
    st.info("Upload an image and click 'Extract LaTeX' to see the results here.")
    uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
                
        # Extract LaTeX button
        if st.button("Extract LaTeX 🔍", type="primary"):
            result = extract_latex_from_image(uploaded_file)
            if result:
                st.session_state['ocr_result'] = result

# Display LaTeX Result
with col2:
    st.header("💡 Output")
    # Main content area: Display the OCR result
    if 'ocr_result' in st.session_state:
        st.markdown("### LaTeX Code")
        st.code(st.session_state['ocr_result'], language='latex')

        st.markdown("### LaTeX Rendered")
        # Remove common delimiters if present (adjust as needed)
        cleaned_latex = st.session_state['ocr_result'].replace(r"\[", "").replace(r"\]", "")
        st.latex(cleaned_latex)

#-------------------------------------
#          Footer
#-------------------------------------
st.markdown("---")
st.markdown("Made using GPT-4 Vision by Rayane Tarkany")
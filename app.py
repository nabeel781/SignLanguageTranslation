import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# -----------------------------
# Custom CSS for Professional Look
# -----------------------------
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #dfe9f3, #ffffff);
    }
    /* Title */
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #2c3e50;
        margin-top: -30px;
        margin-bottom: 10px;
    }
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.3em;
        color: #7f8c8d;
        margin-bottom: 40px;
    }
    /* Uploaded image */
    .uploaded-img {
        border-radius: 20px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
        margin: 20px auto;
    }
    /* Prediction box */
    .prediction-box {
        background: #2ecc71;
        color: white;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.2);
        margin-top: 20px;
    }
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #95a5a6;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_fixed.keras", safe_mode=False)

    # If model has multiple inputs, use only the first
    if len(model.inputs) > 1:
        inp = model.inputs[0]
        out = model.outputs[0]
        model = tf.keras.Model(inputs=inp, outputs=out)

    return model

model = load_model()

# Class names (A-Z + special signs)
class_names = [chr(i) for i in range(65, 91)] + ["del", "nothing", "space"]

# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.title("ℹ️ About This App")
st.sidebar.info(
    """
    This AI-powered **Sign Language Recognition App**  
    detects ASL letters from hand gesture images.  

    📷 **How to use:**  
    - Upload an image of your hand sign.  
    - The model predicts the letter.  
    - See prediction confidence instantly!  
    """
)

st.sidebar.markdown("🔗 Built with [TensorFlow](https://www.tensorflow.org/) & [Streamlit](https://streamlit.io/)")

# -----------------------------
# Main UI
# -----------------------------
st.markdown("<div class='main-title'>🤟 Sign Language Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Empowering communication through AI</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="PNG")

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds) * 100)

    # Stylish prediction result
    st.markdown(
        f"<div class='prediction-box'>Prediction: {pred_class}<br>Confidence: {confidence:.2f}%</div>",
        unsafe_allow_html=True,
    )

    # Confidence bar
    st.progress(int(confidence))

# -----------------------------
# Footer
# -----------------------------
st.markdown("<div class='footer'>✨ Made with ❤️NABEEL BHATTI AI & Data Science Specialist | Machine Learning | Predictive Analytics |</div>", unsafe_allow_html=True)

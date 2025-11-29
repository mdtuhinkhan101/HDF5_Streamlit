import streamlit as st
import numpy as np
import tensorflow as tf

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# ----------------- Page UI -----------------
st.set_page_config(page_title="AI Prediction App")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f5;
    }
    .box {
        max-width: 450px;
        margin: auto;
        background: white;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    .title {
        text-align: center;
        font-weight: 600;
        margin-bottom: 15px;
        color: #333;
        font-size: 26px;
    }
    .btn-custom {
        width: 100%;
        background: #4a6cf7;
        color: white;
        font-size: 18px;
        padding: 10px 0;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .btn-custom:hover {
        background: #2743d8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="box">', unsafe_allow_html=True)
st.markdown('<h3 class="title">AI Prediction Web App</h3>', unsafe_allow_html=True)

# Inputs (8 features)
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)
f5 = st.number_input("Feature 5", value=0.0)
f6 = st.number_input("Feature 6", value=0.0)
f7 = st.number_input("Feature 7", value=0.0)
f8 = st.number_input("Feature 8", value=0.0)

clicked = st.button("Predict")

if clicked:
    try:
        x = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])
        pred = model.predict(x)[0]
        result = int(np.argmax(pred))

        st.success(f"Prediction: {result}")

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)

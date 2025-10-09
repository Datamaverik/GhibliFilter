# app_streamlit.py
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
from PIL import Image
import io
from config import MODEL_PATH, CHECKPOINT_PATH

st.set_page_config(page_title="Real → Ghibli demo", layout="centered")

@st.cache_resource(show_spinner=False)
def load_generator():
    # Try load_model with InstanceNormalization first, else rebuild & restore weights/checkpoint
    try:
        InstanceNormalization = pix2pix.InstanceNormalization
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"InstanceNormalization": InstanceNormalization}, compile=False)
        st.info("Loaded model from .h5")
        return model
    except Exception:
        try:
            model = pix2pix.unet_generator(3, norm_type='instancenorm')
            model.load_weights(MODEL_PATH)
            st.info("Loaded weights into rebuilt generator.")
            return model
        except Exception:
            model = pix2pix.unet_generator(3, norm_type='instancenorm')
            ckpt = tf.train.Checkpoint(generator_g=model)
            ckpt.restore(CHECKPOINT_PATH).expect_partial()
            st.info("Restored generator from checkpoint.")
            return model

def preprocess_image_file(file_bytes):
    img = tf.image.decode_image(file_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # converts to [0,1]
    
    # --- Step 1: Crop to square ---
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    crop_size = tf.minimum(h, w)
    offset_h = (h - crop_size) // 2
    offset_w = (w - crop_size) // 2
    img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, crop_size, crop_size)

    # --- Step 2: Resize to 256x256 ---
    img = tf.image.resize(img, [256, 256])

    # --- Step 3: Normalize to [-1, 1] ---
    img = (img * 2.0) - 1.0

    # --- Step 4: Add batch dimension ---
    return tf.expand_dims(img, 0)


def tensor_to_pil_img(tensor):
    arr = (tf.squeeze(tensor, 0) + 1.0) / 2.0  # [0,1]
    arr = tf.clip_by_value(arr, 0.0, 1.0).numpy()
    arr = (arr * 255).astype('uint8')
    return Image.fromarray(arr)

st.title("Real → Ghibli style (demo)")
st.write("Drag and drop an image file (jpg/png). Model runs on CPU by default if no GPU.")

uploaded_file = st.file_uploader("Drop an image here", type=["png","jpg","jpeg"], accept_multiple_files=False)

generator = load_generator()

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    try:
        input_tensor = preprocess_image_file(file_bytes)
        pred = generator(input_tensor, training=False)

        # input_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        # input_pil = input_pil.resize((256,256))
        input_pil = tensor_to_pil_img(input_tensor)

        output_pil = tensor_to_pil_img(pred)

        col1, col2 = st.columns(2)
        col1.header("Input")
        col1.image(input_pil, use_container_width=True)
        col2.header("Ghibli-styled Output")
        col2.image(output_pil, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to process image: {e}")
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries
from src.disease_info import disease_info
from lime import lime_image
from dotenv import load_dotenv

st.set_page_config(initial_sidebar_state="collapsed")
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .st-emotion-cache-16txtl3 {display: none !important;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)


st.title("🌱 Plant Disease Detection")

load_dotenv()

# Load the model once to avoid multiple reloads
def load_model():
    return tf.keras.models.load_model("trained_model.h5")

model = load_model()

# Model prediction function
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0  # Normalize the image
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
        'Apple___healthy', 'Blueberry___healthy', 
        'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
        'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]
    return class_names[result_index]

def plant_disease_model():
    st.header("🌿 Plant Disease Detection")
    
    st.markdown(
        """
        **Model Information:**
        - 📸 The model is trained on 70,000 images from this [dataset](https://github.com/vam-luffy/dataSet/tree/main/train). 
        - ⚠️ Due to GPU limitations, further training was not performed.
        - ✅ The model provides good accuracy and delivers reliable results for images within this dataset.
        """
    )

    test_image = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, use_container_width=True, caption="Uploaded Image")

        if st.button("Predict"):
            with st.spinner("Processing image..."):
                try:
                    result_class_name = model_prediction(test_image)
                    st.success(f"✅ Prediction: {result_class_name}")
                    st.session_state.prediction_result = result_class_name
                except Exception as e:
                    st.error(f"Error: {e}")

        if "prediction_result" in st.session_state:
            st.write(f"✅ Prediction: {st.session_state.prediction_result}")

        if st.checkbox("Show Model Explanation (LIME)"):
            with st.spinner("Generating explanation..."):
                try:
                    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
                    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
                    input_arr = np.array([input_arr])
                    
                    explainer = lime_image.LimeImageExplainer()
                    explanation = explainer.explain_instance(
                        input_arr[0].astype('double'),
                        model.predict,
                        top_labels=1,
                        hide_color=0,
                        num_samples=1000
                    )

                    temp, mask = explanation.get_image_and_mask(
                        explanation.top_labels[0],
                        positive_only=True,
                        num_features=5,
                        hide_rest=False
                    )

                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(mark_boundaries(temp / 255.0, mask))
                    ax[0].set_title("LIME Explanation")
                    ax[1].imshow(image)
                    ax[1].set_title("Original Image")
                    for a in ax:
                        a.axis("off")
                    st.pyplot(fig)

                    if "prediction_result" in st.session_state:
                        display_disease_info(st.session_state.prediction_result)
                except Exception as e:
                    st.error(f"An error occurred while generating explanation: {e}")

def display_disease_info(class_name):
    info = disease_info.get(class_name, "Disease not found.")
    if info != "Disease not found.":
        st.subheader("Disease Information:")
        for key, value in info.items():
            st.write(f"**{key}:** {value}")
    else:
        st.error("No information available for this disease.")

plant_disease_model()
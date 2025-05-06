import streamlit as st
import numpy as np
import cv2
import pickle
from skimage.feature import hog
from PIL import Image

# --------- Class labels ---------
classes = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]

# --------- Model loader with caching ---------
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "SVM (Linear Kernel)": "D:/New folder/cv/models/svm_linear_model.pkl",
        "SVM (RBF Kernel)": "D:/New folder/cv/models/svm_rbf_model.pkl",
        "K-Nearest Neighbors": "D:/New folder/cv/models/k-nearest_neighbors_model.pkl",
        "Random Forest": "D:/New folder/cv/models/random_forest_model.pkl"
    }
    with open(model_paths[model_name], "rb") as f:
        return pickle.load(f)

# --------- Preprocessing function ---------
def preprocess_image(image, target_size=(128, 128)):
    image = np.array(image)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, target_size)
    return resized

# --------- HOG feature extraction ---------
def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys',
                      visualize=True, feature_vector=True)
    return features

# --------- Streamlit UI ---------
st.title("🩻 Chest X-ray Disease Classification")

# Model selection
model_choice = st.selectbox(
    "Select a model for prediction",
    ["SVM (Linear Kernel)", "SVM (RBF Kernel)", "K-Nearest Neighbors", "Random Forest"]
)

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Chest X-ray", use_column_width=True)

    if st.button("🔍 Predict"):
        model = load_model(model_choice)
        preprocessed = preprocess_image(image)
        features = extract_hog_features(preprocessed).reshape(1, -1)

        prediction = model.predict(features)[0]

        confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = np.max(proba) * 100

        predicted_label = classes[prediction]

        st.success(f"✅ **Predicted Lung Condition:** `{predicted_label}`")
        st.info(f"📈 Confidence: **{confidence:.2f}%**")

        # Optional: Debug info
        with st.expander("🔍 View Debug Info"):
            st.write("Model:", model_choice)
            st.write("Raw class index:", prediction)
            if hasattr(model, "predict_proba"):
                st.write("Class probabilities:")
                for i, p in enumerate(proba):
                    st.write(f"- {classes[i]}: {p*100:.2f}%")

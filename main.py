import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle
import hashlib

import json
import os

# In-memory user "database"
USERS_FILE = "users.json"

# Load users from file
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

# Save users to file
def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# Initialize
users_db = load_users()



# ---------------------- AUTH FUNCTIONS ---------------------- #
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_user(username, password):
    users_db = load_users()
    return username in users_db and users_db[username] == hash_password(password)

def signup_user(username, password):
    users_db = load_users()
    if username in users_db:
        return False
    users_db[username] = hash_password(password)
    save_users(users_db)
    return True



# ---------------------- CAPTION FUNCTION ---------------------- #
# Function to generate and display caption
def generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34,img_size=224):
    # Load the trained models and tokenizer
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Preprocess the image
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    image_features = feature_extractor.predict(img, verbose=0)  # Extract image features

    # Generate the caption
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()

    # Display the image with the generated caption
    img = load_img(image_path, target_size=(img_size, img_size))
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
        
    plt.title(caption, fontsize=16, color='black', fontweight='bold')
    st.pyplot(plt)  # Display image in Streamlit


# ---------------------- MAIN APP UI ---------------------- #
# Streamlit app interface
def main_app():
    st.title("Image Caption Generator")
    st.write("Upload an image and generate a caption using the trained model.")

    # Upload the image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Save the uploaded image temporarily
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Paths for the saved models and tokenizer
        model_path = "models/model.keras"  # Replace with the actual path
        tokenizer_path = "models/tokenizer.pkl"  # Replace with the actual path
        feature_extractor_path = "models/feature_extractor.keras"  # Replace with the actual path

        # Generate caption and display image with caption
        generate_and_display_caption("uploaded_image.jpg", model_path, tokenizer_path, feature_extractor_path)
        
        
        
# ---------------------- LOGIN/SIGNUP PAGE ---------------------- #
def login_signup_page():
    st.title("🔐 Login / Signup")

    page = st.radio("Choose Action", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if page == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.rerun()  # ← add this line to reload app into logged-in state
            else:
                st.error("Invalid username or password.")

    elif page == "Signup":
        if st.button("Signup"):
            if signup_user(username, password):
                st.success("Signup successful. You can now login.")
                st.experimental_set_query_params(page="Login")  # optional redirect
                st.rerun()  # ← add this line to refresh the app after signup
            else:
                st.error("Username already exists.")

        
        
        
# ---------------------- STREAMLIT SESSION FLOW ---------------------- #
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""

    if st.session_state.logged_in:
        st.sidebar.write(f"👤 Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.success("Logged out successfully.")
            st.rerun()  # ← this will re-run and take user back to login
        else:
            main_app()
    else:
        login_signup_page()



if __name__ == "__main__":
    main()
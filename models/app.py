import streamlit as st
import joblib
import os

# Load model and vectorizer
model = joblib.load("models/toxic_comment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("💬 Toxic Comment & Hate Speech Detection")
st.markdown("Detects whether a comment is toxic or clean (non-toxic).")

# Input method selection
option = st.radio("Select input method:", ["Single Text", "Upload .txt File"])

if option == "Single Text":
    user_input = st.text_area("Enter your comment here:")
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            vec_text = vectorizer.transform([user_input])
            prediction = model.predict(vec_text)[0]
            label = "🟩 Clean" if prediction == 0 else "🟥 Toxic"
            st.markdown(f"### Result: **{label}**")

elif option == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        lines = uploaded_file.read().decode("utf-8").splitlines()
        if len(lines) == 0:
            st.warning("The file is empty!")
        else:
            st.markdown("### Results:")
            vec_lines = vectorizer.transform(lines)
            predictions = model.predict(vec_lines)
            for i, line in enumerate(lines):
                label = "🟩 Clean" if predictions[i] == 0 else "🟥 Toxic"
                st.write(f"{i+1}. **{label}** → {line}")

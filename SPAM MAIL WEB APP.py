
import pickle
import streamlit as st
import base64


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Spam Mail Prediction",
    page_icon="📧",
    layout="centered"
)


# --------------------------------------------------
# File paths
# --------------------------------------------------





BACKGROUND_PATH = r"D:\projects ml\SPAM MAIL PREDICTION\SPAM MAIL BACKGROUND.png"


# --------------------------------------------------
# Load model
# --------------------------------------------------

load_model=pickle.load(open("D:/projects ml/SPAM MAIL PREDICTION/spam_mail_prediction.sav","rb"))


# --------------------------------------------------
# Load TF-IDF vectorizer
# --------------------------------------------------

feature_extraction=pickle.load(open("D:/projects ml/SPAM MAIL PREDICTION/features_extraction.sav","rb"))


# --------------------------------------------------
# Background image
# --------------------------------------------------

with open(BACKGROUND_PATH, "rb") as file:
    encoded_image = base64.b64encode(file.read()).decode()


# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown(
    f"""
    <style>

    .stApp {{
        background-image: url(
            "data:image/png;base64,{encoded_image}"
        );

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .block-container {{
        background-color: rgba(255, 255, 255, 0.94);
        padding: 40px;
        border-radius: 20px;
        max-width: 850px;
        margin-top: 40px;
        box-shadow: 0px 8px 30px rgba(0,0,0,0.25);
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def spam_mail_pred(input_mail):

    input_features = feature_extraction.transform(
        [input_mail]
    )

    prediction = load_model.predict(input_features)

    if prediction[0] == 0:

        st.error("🚨 SPAM MAIL")

        st.write(
            "This email appears to be a spam message."
        )

    else:

        st.success("✅ HAM MAIL")

        st.write(
            "This email appears to be a legitimate email."
        )


# --------------------------------------------------
# Main application
# --------------------------------------------------

def main():

    st.title("📧 SPAM MAIL PREDICTION")

    st.write(
        "Enter an email message below to check whether "
        "it is Spam or Ham."
    )

    input_mail = st.text_area(
        "📩 Enter Email Content",
        height=250,
        placeholder=(
            "Example:\n\n"
            "Congratulations! You have won a free iPhone. "
            "Click here now!"
        )
    )

    if st.button("🔍 Predict Email"):

        if input_mail.strip() == "":
            st.warning("⚠️ Please enter an email.")

        else:
            spam_mail_pred(input_mail)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
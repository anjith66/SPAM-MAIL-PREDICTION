```python
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

BACKGROUND_PATH = "SPAM MAIL BACKGROUND.png"


# --------------------------------------------------
# Load model
# --------------------------------------------------

load_model = pickle.load(
    open("spam_mail_prediction.sav", "rb")
)


# --------------------------------------------------
# Load TF-IDF vectorizer
# --------------------------------------------------

feature_extraction = pickle.load(
    open("features_extraction.sav", "rb")
)


# --------------------------------------------------
# Background image
# --------------------------------------------------

with open(BACKGROUND_PATH, "rb") as file:
    encoded_image = base64.b64encode(
        file.read()
    ).decode()


# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown(
    f"""
    <style>

    /* Background */
    .stApp {{
        background-image: url(
            "data:image/png;base64,{encoded_image}"
        );

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}


    /* Main white container */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.94);
        padding: 40px;
        border-radius: 20px;
        max-width: 850px;
        margin-top: 40px;

        box-shadow:
            0px 8px 30px rgba(0, 0, 0, 0.25);
    }}


    /* Main title */
    .main-title {{
        color: black !important;
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 15px;
    }}


    /* Description */
    .description {{
        color: black !important;
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }}


    /* Text area label */
    label {{
        color: black !important;
        font-weight: 600 !important;
    }}


    /* Predict button */
    .stButton > button {{
        width: 100%;
        height: 50px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
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

    prediction = load_model.predict(
        input_features
    )

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

    # Black title
    st.markdown(
        """
        <div class="main-title">
            📧 SPAM MAIL PREDICTION
        </div>
        """,
        unsafe_allow_html=True
    )


    # Description
    st.markdown(
        """
        <div class="description">
            Enter an email message below to check whether
            it is Spam or Ham.
        </div>
        """,
        unsafe_allow_html=True
    )


    # Email input
    input_mail = st.text_area(
        "📩 Enter Email Content",
        height=250,
        placeholder=(
            "Example:\n\n"
            "Congratulations! You have won a free iPhone. "
            "Click here now!"
        )
    )


    # Prediction button
    if st.button("🔍 Predict Email"):

        if input_mail.strip() == "":

            st.warning(
                "⚠️ Please enter an email."
            )

        else:

            spam_mail_pred(input_mail)


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    main()
```

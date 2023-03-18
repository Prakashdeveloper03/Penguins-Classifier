import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Sets page's title and icon
st.set_page_config(page_title="Penguins Classifier", page_icon="🎯")

# Sets the title of the page
st.title("Penguins Classifier")

# Creates a sidebar header for user input
st.sidebar.header("User Input Features")

# Collects user input features into a dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    # Defines a function to collect user input through the sidebar widgets
    def user_input_features():
        island = st.sidebar.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
        sex = st.sidebar.selectbox("Sex", ("male", "female"))
        bill_length_mm = st.sidebar.slider("Bill length (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider(
            "Flipper length (mm)", 172.0, 231.0, 201.0
        )
        body_mass_g = st.sidebar.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)
        data = {
            "island": island,
            "bill_length_mm": bill_length_mm,
            "bill_depth_mm": bill_depth_mm,
            "flipper_length_mm": flipper_length_mm,
            "body_mass_g": body_mass_g,
            "sex": sex,
        }
        return pd.DataFrame(data, index=[0])


# Calls the user_input_features() function to get input values
input_df = user_input_features()

# Reads the cleaned penguins dataset
penguins_raw = pd.read_csv("data/penguins.csv")
penguins = penguins_raw.drop(columns=["species"])
df = pd.concat([input_df, penguins], axis=0)

# One-hot encodes categorical features
encode = ["sex", "island"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]

# Displays the user input features
st.subheader("User Input features")
if uploaded_file is None:
    st.write(
        "Awaiting CSV file to be uploaded. Currently using example input parameters (shown below)."
    )
st.write(df)

# Loads the saved classification model
load_clf = joblib.load("model/penguins_clf.pkl")

# Uses the loaded model to make predictions on the user input
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# Displays the predicted penguin species
st.subheader("Prediction")
penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
st.write(penguins_species[prediction])

# Displays the predicted probability for each penguin species
st.subheader("Prediction Probability")
st.write(prediction_proba)

# 1 is good (Lower Risk)
# 0 is bad (Higher Risk)

import streamlit as st
import pandas as pd
import joblib as jb


model = jb.load("models/extra_tree_credit_model.pkl")
encoders = {
    col: jb.load(f'models/{col}.pkl') for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

st.title("Prediction sur les risques de credit")
st.write("Rentrez les information du client pour verifier si les risques sont bon ou mauvais")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", 'female'])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Account", ["little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (mounths)", min_value=1, value=12)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders['Sex'].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders['Saving accounts'].transform([saving_accounts])[0]],
    "Checking account": [encoders['Checking account'].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

if st.button("prediction"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("La prédiction est : **BONNE** (Risque Faible) ✅")
        st.balloons()  # Petit effet de fête !
    else:
        st.error("La prédiction est : **MAUVAISE** (Risque Élevé) ⚠️")

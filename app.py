# 1 is good (Lower Risk)
# 0 is bad (Higher Risk)

import streamlit as st
import pandas as pd
import joblib as jb

# Configuration de la page
st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

# Charger Tailwind CSS
st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
    body { font-family: 'Inter', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# Chargement du modèle et des encodeurs
@st.cache_resource
def load_resources():
    model = jb.load("models/extra_tree_credit_model.pkl")
    encoders = {
        col: jb.load(f'models/{col}.pkl') for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
    }
    return model, encoders

model, encoders = load_resources()

# --- HEADER ---
st.markdown("""
    <div class="bg-gradient-to-r from-purple-600 to-blue-600 text-white py-12 px-6 rounded-lg mb-8">
        <h1 class="text-4xl font-bold mb-2">💳 Analyse de Risque de Crédit</h1>
        <p class="text-lg opacity-90">Évaluez la fiabilité d'un dossier client instantanément</p>
    </div>
""", unsafe_allow_html=True)

# --- FORM CONTAINER ---
st.markdown('<div class="bg-white rounded-lg shadow-lg p-8 mb-8">', unsafe_allow_html=True)

# Colonnes
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h2 class="text-2xl font-bold text-gray-800 mb-6">👤 Infos Personnelles</h2>', unsafe_allow_html=True)
    age = st.number_input("Âge", 18, 80, 30)
    sex = st.selectbox("Sexe", ["male", "female"], key="sex")
    job = st.number_input("Niveau d'emploi (0-3)", 0, 3, 1)
    housing = st.selectbox("Logement", ["own", "rent", "free"], key="housing")

with col2:
    st.markdown('<h2 class="text-2xl font-bold text-gray-800 mb-6">💰 Infos Financières</h2>', unsafe_allow_html=True)
    saving_accounts = st.selectbox("Compte Épargne", ["little", "moderate", "rich", "quite rich"], key="saving")
    checking_account = st.selectbox("Compte Courant", ["little", "moderate", "rich"], key="checking")
    credit_amount = st.number_input("Montant ($)", min_value=0, value=1000)
    duration = st.number_input("Durée (mois)", min_value=1, value=12)

st.markdown('</div>', unsafe_allow_html=True)

# --- BUTTON ---
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    if st.button("🔍 Analyser le Risque", type="primary", use_container_width=True):
        # Construction du DataFrame avec les encodeurs
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

        # Prédiction avec spinner
        with st.spinner('⏳ Analyse en cours...'):
            pred = model.predict(input_df)[0]

        # Affichage des résultats
        if pred == 1:
            st.markdown("""
                <div class="bg-gradient-to-r from-green-400 to-green-600 text-white p-8 rounded-lg mt-8 shadow-lg">
                    <h2 class="text-3xl font-bold mb-2">✅ RISQUE FAIBLE</h2>
                    <p class="text-lg">Le dossier présente un RISQUE FAIBLE. Le crédit peut être accordé.</p>
                    <div class="mt-4 pt-4 border-t border-green-300">
                        <p class="text-base"><strong>Score de Risque:</strong> Favorable</p>
                        <p class="text-base"><strong>Confiance:</strong> Élevée</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("""
                <div class="bg-gradient-to-r from-red-500 to-red-700 text-white p-8 rounded-lg mt-8 shadow-lg">
                    <h2 class="text-3xl font-bold mb-2">⚠️ RISQUE ÉLEVÉ</h2>
                    <p class="text-lg">Le modèle détecte une probabilité de défaut importante.</p>
                    <div class="mt-4 pt-4 border-t border-red-400">
                        <p class="text-base"><strong>Score de Risque:</strong> Défavorable</p>
                        <p class="text-base"><strong>Risque de défaut:</strong> Probable</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
    <div class="text-center text-gray-500 mt-12 pt-8 border-t border-gray-200">
        <p class="text-sm">Système de Prédiction de Risque de Crédit - Powered by Machine Learning 🤖</p>
    </div>
""", unsafe_allow_html=True)
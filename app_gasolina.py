import numpy as np
import streamlit as st
import pandas as pd
import joblib

st.write('''# Aplicación para predecir el costo de gasolina en México <br/> G. de la Cruz A.''')
st.image("img1.jpg", caption="se empleo la base en datos de la CRE", width=450)

st.header('Parametros para la predicción')

# --- Cargar modelo y encoder ---
modelo = joblib.load("modelo_gasolina.joblib")
encoder = joblib.load("encoder_gasolina.joblib")

# --- Cargar dataset tidy para lista de estados ---
df = pd.read_csv("precios_gasolina_tidy.csv")
estados = df['estado'].unique()

# --- Entradas del usuario ---
estado = st.selectbox('Estado:', estados)
año = st.number_input('Año:', min_value=2017, max_value=2030, value=2023, step=1)
mes = st.number_input('Mes:', min_value=1, max_value=12, value=1, step=1)

# --- Transformar entrada ---
estado_encoded = encoder.transform([[estado]]).toarray()
entrada = np.concatenate([estado_encoded, [[año, mes]]], axis=1)

# --- Predicción ---
prediccion = modelo.predict(entrada)

st.subheader('Precio estimado de gasolina')
st.write(f" **${prediccion[0]:.2f} MXN por litro**")


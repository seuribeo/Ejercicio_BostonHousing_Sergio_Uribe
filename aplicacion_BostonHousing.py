import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import pandas as pd

# Cargar el dataset de Boston desde OpenML
def load_boston_data():
    """Cargar el conjunto de datos de Boston Housing desde OpenML."""
    boston = fetch_openml(name='boston', version=1)
    return boston

# Función para cargar el modelo entrenado
def load_model():
    """Cargar el modelo preentrenado."""
    with open('model_trained_regressor.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Preprocesamiento de las entradas de texto
def preprocess_input(input_data):
    """Transforma los datos de entrada en un formato adecuado para la predicción."""
    # Convertir las entradas a un arreglo numpy
    input_array = np.array([input_data])
    # Normalizar las entradas utilizando el mismo scaler usado durante el entrenamiento
    scaler = StandardScaler()
    input_array = scaler.fit_transform(input_array)
    return input_array

# Página principal de la aplicación
def main():
    # Estilos personalizados
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 32px;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
        }
        .description {
            font-size: 18px;
            color: #555555;
            text-align: center;
            margin-bottom: 20px;
        }
        .footer {
            font-size: 14px;
            color: #888888;
            text-align: center;
            margin-top: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Título y descripción
    st.markdown('<div class="main-title">Predicción del Precio de una Casa (Boston Housing)</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Introduce las características de una casa y te diremos su precio aproximado.</div>', unsafe_allow_html=True)

    # Cargar el dataset de Boston Housing
    boston = load_boston_data()
    feature_names = boston.feature_names

    # Solicitar las entradas de texto del usuario
    st.subheader("Introduce las características de la casa:")
    inputs = {}
    for feature in feature_names:
        inputs[feature] = st.number_input(f'{feature}', min_value=0.0)

    # Lista con las características de la casa
    house_features = [inputs[feature] for feature in feature_names]

    # Cuando el usuario hace clic en el botón de predicción
    if st.button('Predecir Precio'):
        with st.spinner("Calculando el precio..."):
            # Cargar el modelo entrenado
            model = load_model()

            # Preprocesar las entradas del usuario
            input_data = preprocess_input(house_features)

            # Realizar la predicción
            prediction = model.predict(input_data)

            # Mostrar el resultado de la predicción
            st.success(f"El precio estimado de la casa es: ${prediction[0]:,.2f}")

    # Footer
    st.markdown('<div class="footer">© 2025 - Predicción de precio de casas con Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()



import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
import pandas as pd

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

    # Solicitar las entradas de texto del usuario
    st.subheader("Introduce las características de la casa:")
    CRIM = st.number_input('Tasa de criminalidad per cápita', min_value=0.0)
    ZN = st.number_input('Proporción de terrenos residenciales', min_value=0.0)
    INDUS = st.number_input('Proporción de áreas comerciales no minoristas', min_value=0.0)
    CHAS = st.number_input('¿Está cerca del río Charles? (1 = sí, 0 = no)', min_value=0, max_value=1)
    NOX = st.number_input('Concentración de óxidos de nitrógeno (partes por 10 millones)', min_value=0.0)
    RM = st.number_input('Número promedio de habitaciones por vivienda', min_value=0.0)
    AGE = st.number_input('Proporción de viviendas construidas antes de 1940', min_value=0.0)
    DIS = st.number_input('Distancia ponderada a los 5 centros de empleo', min_value=0.0)
    RAD = st.number_input('Índice de accesibilidad a las autopistas', min_value=0.0)
    TAX = st.number_input('Tasa de impuestos sobre la propiedad', min_value=0.0)
    PTRATIO = st.number_input('Relación alumno-profesor', min_value=0.0)
    B = st.number_input('Proporción de población de origen afroamericano', min_value=0.0)
    LSTAT = st.number_input('Porcentaje de población con bajo estatus socioeconómico', min_value=0.0)

    # Lista con las características de la casa
    house_features = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]

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


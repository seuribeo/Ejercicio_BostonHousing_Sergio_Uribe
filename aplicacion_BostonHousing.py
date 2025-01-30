import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar el modelo previamente entrenado
@st.cache
def load_model():
    """Carga el modelo de regresión preentrenado."""
    with open('model_trained_regressor.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Función para preprocesar la entrada de los datos
def preprocess_input(features):
    """Preprocesa los datos de entrada para que sean compatibles con el modelo."""
    # Aseguramos que las características están en el formato adecuado para el modelo
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(np.array(features).reshape(1, -1))
    return features_scaled

def main():
    # Título y descripción
    st.title('Predicción del precio de casas - Boston Housing')
    st.markdown("""
    Introduce las características de una casa para predecir su precio en el mercado. 
    El modelo está entrenado sobre el conjunto de datos Boston Housing.
    """)

    # Definir los campos para las características de la casa (13 en total)
    st.subheader("Características de la casa:")
    
    # Crear entradas de texto para las 13 características
    crim = st.number_input('Tasa de criminalidad (CRIM)', min_value=0.0, value=0.1)
    zn = st.number_input('Porcentaje de terrenos residenciales zonificados (ZN)', min_value=0.0, value=20.0)
    indus = st.number_input('Porcentaje de terrenos comerciales (INDUS)', min_value=0.0, value=5.0)
    chas = st.selectbox('¿Está cerca de un río? (CHAS)', options=[0, 1])
    nox = st.number_input('Concentración de óxidos nítricos (NOX)', min_value=0.0, value=0.5)
    rm = st.number_input('Número promedio de habitaciones por vivienda (RM)', min_value=0.0, value=6.0)
    age = st.number_input('Proporción de viviendas construidas antes de 1940 (AGE)', min_value=0.0, value=40.0)
    dis = st.number_input('Distancia a centros de empleo (DIS)', min_value=0.0, value=4.0)
    rad = st.number_input('Índice de accesibilidad a vías radiales (RAD)', min_value=0, value=5)
    tax = st.number_input('Tasa de impuestos sobre la propiedad (TAX)', min_value=0, value=300)
    ptratio = st.number_input('Proporción alumnos-profesor (PTRATIO)', min_value=10, value=18)
    b = st.number_input('Índice de proporción de población afroamericana (B)', min_value=0.0, value=350.0)
    lstat = st.number_input('Porcentaje de población con nivel socioeconómico bajo (LSTAT)', min_value=0.0, value=10.0)
    
    # Recolectar las características en una lista
    features = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]

    # Cargar el modelo
    model = load_model()

    # Botón para predecir el precio de la casa
    if st.button("Predecir Precio"):
        # Preprocesar los datos
        features_scaled = preprocess_input(features)
        
        # Realizar la predicción
        price_pred = model.predict(features_scaled)
        
        # Mostrar el precio predicho
        st.write(f"El precio predicho de la casa es: ${price_pred[0]:,.2f}")

        # Mostrar los hiperparámetros del modelo
        st.subheader("Hiperparámetros del mejor modelo:")
        st.write("""
        - **n_estimators**: 100
        - **max_depth**: 10
        - **min_samples_split**: 2
        - **min_samples_leaf**: 1
        """)

    # Pie de página
    st.markdown("### © 2025 - Predicción de precio de casas")

if __name__ == "__main__":
    main()


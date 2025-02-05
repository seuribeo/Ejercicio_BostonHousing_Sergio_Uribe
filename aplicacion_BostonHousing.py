import streamlit as st
import pickle
import gzip
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar el modelo
@st.cache_data
def load_model():
    filename = 'model_trained_regressor.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Función para realizar predicciones
def predict_price(features):
    model = load_model()
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

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
st.markdown('<div class="main-title">Predicción de Precios de Vivienda en Boston</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Ingrese los valores de las características para predecir el precio de una vivienda.</div>', unsafe_allow_html=True)

# Entrada de características (fuera de la barra lateral)
st.header("Parámetros de Entrada")
crim = st.number_input("Tasa de criminalidad", min_value=0.0, value=0.1)
zn = st.number_input("Porcentaje de terrenos residenciales", min_value=0.0, value=10.0)
indus = st.number_input("Porcentaje de acres comerciales", min_value=0.0, value=5.0)
nox = st.number_input("Concentración de óxidos de nitrógeno", min_value=0.0, max_value=1.0, value=0.5)
rm = st.number_input("Número promedio de habitaciones", min_value=1.0, value=6.0)
age = st.number_input("Edad promedio de las viviendas", min_value=0.0, max_value=100.0, value=50.0)
dis = st.number_input("Distancia a centros de empleo", min_value=0.0, value=4.0)
rad = st.number_input("Índice de accesibilidad a carreteras", min_value=1, max_value=24, value=4)
tax = st.number_input("Tasa de impuesto a la propiedad", min_value=0.0, value=300.0)
ptratio = st.number_input("Ratio de alumnos por maestro", min_value=0.0, value=18.0)
lstat = st.number_input("Porcentaje de población de bajos ingresos", min_value=0.0, value=12.0)

# Botón de predicción (fuera de la barra lateral)
if st.button("Predecir Precio"):
    features = [crim, zn, indus, nox, rm, age, dis, rad, tax, ptratio, lstat]
    prediction = predict_price(features)
    st.success(f"El precio estimado de la vivienda es: ${prediction * 1000:,.2f}")

# Información sobre los hiperparámetros evaluados (barra lateral)
st.sidebar.header("Hiperparámetros Evaluados")
st.sidebar.markdown(""" 
Se probaron diferentes modelos con diversas configuraciones de hiperparámetros. Los principales modelos evaluados fueron:

- **ElasticNet con StandardScaler** (Mejor MAE: 3.4372)
- **Kernel Ridge con StandardScaler** (Mejor MAE: 2.6156, modelo seleccionado)
- **ElasticNet con MinMaxScaler** (Mejor MAE: 3.4694)
- **Kernel Ridge con MinMaxScaler** (Mejor MAE: 2.8787)

El modelo seleccionado fue **Kernel Ridge con StandardScaler**, ya que presentó el menor MAE.
""")

# Footer
st.markdown('<div class="footer">© 2025 - Predicción de precios con Streamlit</div>', unsafe_allow_html=True)


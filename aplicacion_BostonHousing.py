import streamlit as st
import pickle
import numpy as np

# Función para cargar el modelo preentrenado
def load_model():
    """Carga el modelo preentrenado con el mejor ajuste encontrado."""
    with open('model_trained_regressor.pkl', 'rb') as f:
        model = pickle.load(f)  # Puede ser un Pipeline con StandardScaler + Kernel Ridge
    return model

# Cargar el modelo una sola vez al inicio
model = load_model()

# Hiperparámetros óptimos encontrados en la búsqueda del profesor
best_model_name = "Kernel Ridge"
best_scaler = "StandardScaler"
best_hyperparams = {
    "alpha": 0.1,
    "kernel": "rbf"
}

# Función principal de la aplicación Streamlit
def main():
    # Estilos personalizados
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 32px;
            font-weight: bold;
            color: #000000;
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

    # Barra lateral con los hiperparámetros evaluados
    st.sidebar.header("Hiperparámetros Evaluados")
    st.sidebar.markdown(""" 
    Se probaron diferentes modelos con diversas configuraciones de hiperparámetros. Los principales modelos evaluados fueron:

    - *ElasticNet con StandardScaler* (Mejor MAE: 3.4372)
    - *Kernel Ridge con StandardScaler* (Mejor MAE: 2.6156, modelo seleccionado)
    - *ElasticNet con MinMaxScaler* (Mejor MAE: 3.4694)
    - *Kernel Ridge con MinMaxScaler* (Mejor MAE: 2.8787)

    El modelo seleccionado fue *Kernel Ridge con StandardScaler*, ya que presentó el menor MAE.
    """)

    # Título de la aplicación
    st.markdown('<div class="main-title">Predicción del Precio de una Casa - Boston Housing</div>', unsafe_allow_html=True)

    # Descripción del modelo
    st.markdown(f"""
    ### Modelo seleccionado:
    - *Regresor:* {best_model_name}
    - *Escalador:* {best_scaler}
    - *Mejores hiperparámetros:*  
        - α (alpha): {best_hyperparams['alpha']}  
        - Kernel: {best_hyperparams['kernel']}
    """)

    # Sección de entrada de características
    st.subheader("Introduce las características de la casa:")
    
    # Entradas para las 13 características
    CRIM = st.number_input("CRIM - Tasa de criminalidad", value=0.1)
    ZN = st.number_input("ZN - Proporción de terrenos residenciales zonificados", value=0.0)
    INDUS = st.number_input("INDUS - Proporción de acres de negocios no minoristas", value=10.0)
    CHAS = st.number_input("CHAS - Proximidad al río Charles (0 o 1)", value=0)
    NOX = st.number_input("NOX - Concentración de óxidos de nitrógeno", value=0.5)
    RM = st.number_input("RM - Número promedio de habitaciones", value=6.0)
    AGE = st.number_input("AGE - Proporción de casas antiguas", value=50.0)
    DIS = st.number_input("DIS - Distancia a centros de empleo", value=5.0)
    RAD = st.number_input("RAD - Índice de accesibilidad a carreteras", value=4)
    TAX = st.number_input("TAX - Tasa de impuestos", value=300)
    PTRATIO = st.number_input("PTRATIO - Relación alumno/profesor", value=18)
    B = st.number_input("B - Proporción de residentes afroamericanos", value=400)
    LSTAT = st.number_input("LSTAT - Porcentaje de población de bajo estatus", value=12.0)

    # Crear un array con las características ingresadas
    features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

    # Botón para predecir el precio
    if st.button("Predecir precio de la casa"):
        # Realizar la predicción usando el modelo cargado
        predicted_price = model.predict(features)[0]

        # Mostrar el resultado
        st.success(f"💰 *El precio estimado de la casa es: ${predicted_price:,.2f}*")

    # Footer
    st.markdown('<div class="footer">© 2025 - Predicción de precios con Streamlit</div>', unsafe_allow_html=True)

if _name_ == "_main_":
    main()

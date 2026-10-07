
import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load("iris_decision_tree.pkl")

# Load Iris dataset for feature names and class names
iris = load_iris()

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classifier")

st.write(
    "Enter the flower measurements below to predict "
    "the Iris flower species using a Decision Tree model."
)

st.subheader("Enter Flower Measurements")

# Input sliders
sepal_length = st.slider(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.1,
    step=0.1
)

sepal_width = st.slider(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=4.5,
    value=3.5,
    step=0.1
)

petal_length = st.slider(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=1.4,
    step=0.1
)

petal_width = st.slider(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=2.5,
    value=0.2,
    step=0.1
)

# Prediction button
if st.button("Predict Flower"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)

    predicted_class = iris.target_names[prediction[0]]

    st.success(
        f"Predicted Flower: {predicted_class.capitalize()}"
    )

    st.write("### Input Values")

    st.write(f"Sepal Length: {sepal_length} cm")
    st.write(f"Sepal Width: {sepal_width} cm")
    st.write(f"Petal Length: {petal_length} cm")
    st.write(f"Petal Width: {petal_width} cm")

st.write("---")
st.write("Built using Python, Scikit-Learn, Joblib and Streamlit.")

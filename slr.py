import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# Streamlit App Title
st.title("Simple Linear Regression Prediction App")

# Step 1: Upload the CSV file
st.sidebar.header("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

# Step 2: Load and display the CSV data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Check if there are at least two columns for X and Y
    if data.shape[1] >= 2:
        # Step 3: Get X (independent) and Y (dependent) variables
        st.sidebar.header("Configure X and Y")
        x_col = st.sidebar.selectbox("Select independent variable (X):", data.columns)
        y_col = st.sidebar.selectbox("Select dependent variable (Y):", data.columns)

        # Step 4: Train the model using Linear Regression
        X = data[[x_col]]
        Y = data[y_col]
        
        # Train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Create a Linear Regression model and fit it to the data
        model = LinearRegression()
        model.fit(X_train, Y_train)
        
        # Step 5: Make Predictions
        st.subheader("Make Predictions")
        input_value = st.number_input(f"Enter a value for {x_col}:", min_value=float(X.min()), max_value=float(X.max()))
        
        # Predict the corresponding value of Y for the input X
        if input_value:
            prediction = model.predict(np.array([[input_value]]))
            st.write(f"Predicted value of {y_col}: {prediction[0]}")

        # Step 6: Display the model's performance
        st.subheader("Model Performance")
        st.write(f"Model's R^2 score on test data: {model.score(X_test, Y_test):.4f}")
        
        # Step 7: Visualize the data and regression line
        st.subheader("Visualization")
        
        # Plotting the regression line
        plt.figure(figsize=(8, 6))
        sns.regplot(x=x_col, y=y_col, data=data, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
        plt.title("Linear Regression Line")
        st.pyplot(plt)
    
    else:
        st.warning("The dataset does not have enough columns to perform linear regression.")

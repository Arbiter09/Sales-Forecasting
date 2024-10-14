import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from streamlit_extras.colored_header import colored_header

def app():
    colored_header(
        label='Welcome to Data :red[Prediction] page 👋🏼',
        color_name='red-70',
        description='Weekly Sales Prediction for the following year'
    )

    @st.cache_data
    def data():
        df = pd.read_csv('C:\\Users\\shahj\\OneDrive\\Desktop\\Projects\\Final-Retail-Sales-Forecasting-main\\Cleaned_Store_data2.csv')
  # Ensure the path is correct
        return df

    df = data()
    
    x = df.drop(['Size', 'Type', 'Date', 'weekly_sales'], axis=1)

    with st.form(key='form', clear_on_submit=False):
        # Form inputs for the prediction
        store = st.selectbox("**Select a Store**", options=df['Store'].unique())
        dept = st.selectbox("**Select a Department**", options=df['Dept'].unique())
        holiday = st.radio("**Click Holiday is True or False**", options=[True, False], horizontal=True)
        temperature = st.number_input(f"**Enter a Temperature (Min: {df['Temperature'].min()}, Max: {df['Temperature'].max()})**")
        fuel = st.number_input(f"**Enter a Fuel Price (Min: {df['Fuel_Price'].min()}, Max: {df['Fuel_Price'].max()})**")
        cpi = st.number_input(f"**Enter a Customer Price Index (Min: {df['CPI'].min()}, Max: {df['CPI'].max()})**")
        unemployment = st.number_input(f"**Enter Unemployment (Min: {df['Unemployment'].min()}, Max: {df['Unemployment'].max()})**")
        year = st.selectbox("**Select a Year**", options=[2010, 2011, 2012, 2013, 2014])
        yearofweek = st.selectbox("**Select Year of Week**", options=df['week_of_year'].unique())
        markdown = st.number_input(f"**Enter a Markdown (Min: {df['Markdown'].min()}, Max: {df['Markdown'].max()})**")

        def inv_trans(x):
            return x if x == 0 else 1 / x

        def is_holiday(x):
            return 1 if x else 0

        button = st.form_submit_button("Predict")

        if button:
            try:
                # Set the tracking URI for MLflow
                mlflow.set_tracking_uri("http://127.0.0.1:5000")
                mlflow.set_experiment("Retail Sales Forecasting")
                
                # Load the model
                model_uri = "runs:/adac4e30f5394197af327fef9daddf7b/model"
                model = mlflow.sklearn.load_model(model_uri)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return  # Stop further execution

            # Check if model was successfully loaded before calling model.predict()
            if model:
                try:
                    result = model.predict([[store, dept, holiday, temperature, fuel, cpi, unemployment, year, yearofweek, inv_trans(markdown)]])
                    st.markdown(f"## Predicted Weekly Sales: {result[0]}")
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
            else:
                st.error("Model is not loaded. Please check for errors in loading the model.")


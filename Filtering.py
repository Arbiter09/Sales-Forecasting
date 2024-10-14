import streamlit as st
import pandas as pd
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.colored_header import colored_header

def app():
    colored_header(
        label='You are in Data :blue[Filtering] page',
        color_name='blue-70',
        description=''
    )

    @st.cache_data
    def load_data():
        df = pd.read_csv('C:\\Users\\shahj\\OneDrive\\Desktop\\Projects\\Final-Retail-Sales-Forecasting-main\\Cleaned_Store_data2.csv')
        # Ensure 'Date' column is converted to datetime
        df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors='coerce')
        
        # Create 'year_of_week' and 'day_of_week' columns
        df['year_of_week'] = df['Date'].dt.isocalendar().week
        df['day_of_week'] = df['Date'].dt.dayofweek  # 0 = Monday, 6 = Sunday

        # Map the day of the week
        df['day_of_week'] = df['day_of_week'].map({
            0: 'Monday', 
            1: 'Tuesday', 
            2: 'Wednesday', 
            3: 'Thursday', 
            4: 'Friday', 
            5: 'Saturday', 
            6: 'Sunday'
        })
        
        return df

    df = load_data()
    filter = dataframe_explorer(df)

    button = st.button('**SUBMIT**', use_container_width=True)
    if button:
        st.dataframe(filter, use_container_width=True, hide_index=True)
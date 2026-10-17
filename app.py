import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title='Delivery Delay Predictor', layout='centered')

st.title('🚚 Delivery Delay Predictor')
st.write('Enter the details below to predict if your delivery will be delayed.')

# Try to load the trained Decision Tree model
try:
    model = joblib.load('decision_tree_model.joblib')
except Exception as e:
    st.error(f'Error loading model: {e}')

# Create input form for user features
with st.form('prediction_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        delivery_distance = st.number_input('Delivery Distance', min_value=0.0, max_value=500.0, value=20.0, step=1.0)
        traffic_congestion = st.slider('Traffic Congestion Score', min_value=1, max_value=5, value=3)
        weather_condition = st.slider('Weather Condition Score', min_value=1, max_value=5, value=3)
        delivery_slot = st.slider('Delivery Slot', min_value=1, max_value=4, value=2)
        driver_experience = st.number_input('Driver Experience (Years)', min_value=0, max_value=50, value=5)
        num_stops = st.number_input('Number of Stops', min_value=0, max_value=20, value=3)
        
    with col2:
        vehicle_age = st.number_input('Vehicle Age (Years)', min_value=0, max_value=30, value=5)
        road_condition = st.slider('Road Condition Score', min_value=1, max_value=5, value=3)
        package_weight = st.number_input('Package Weight (kg)', min_value=0.0, max_value=100.0, value=15.0)
        fuel_efficiency = st.number_input('Fuel Efficiency (mpg)', min_value=0.0, max_value=50.0, value=15.0)
        warehouse_time = st.number_input('Warehouse Processing Time (mins)', min_value=0, max_value=300, value=45)

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Construct input dataframe with precise feature order
    input_features = pd.DataFrame([[
        delivery_distance, traffic_congestion, weather_condition,
        delivery_slot, driver_experience, num_stops, vehicle_age,
        road_condition, package_weight, fuel_efficiency, warehouse_time
    ]], columns=[
        'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
        'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
        'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
        'Warehouse_Processing_Time'
    ])
    
    try:
        prediction = model.predict(input_features)[0]
        
        if prediction == 1:
            st.error('⚠️ Prediction: The delivery is highly likely to be DELAYED.')
        else:
            st.success('✅ Prediction: The delivery is on schedule (NO DELAY).')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')

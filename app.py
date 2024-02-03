import streamlit as st
import pandas as pd
import numpy as np
import joblib

#import files

rf = joblib.load('pipe.joblib')

# read in data
data = pd.read_csv('traineddata.csv')

data['IPS'].unique()

#title
st.title('Laptop Price Predictor in Naira')

## select brand or company name in a dropdown box

company = st.selectbox('Brand', data['Company'].unique())

## select the type of laptop in a dropdown box

type = st.selectbox('Type', data['TypeName'].unique())

# select ram in the laptop

ram = st.selectbox('Ram (GB)', [2,4,6,8,12,16,24,32,64] )

#Select OS of laptop

os = st.selectbox('OS', data['OpSystem'].unique())

# input weight of laptop

weight = st.number_input('Enter the weight of the laptop (in kg)', min_value=0.5, max_value=4.0)

if weight:
    st.write(f'You entered {weight} kg')
else:
    st.write('Please enter a weight to proceed')
    
#weight = st.number_input('Input weight of laptop in Kg')

# select if you want touchscreen or not

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# select IPS

ips = st.selectbox('IPS', ['No', 'Yes'])

# input screen size

screen_size = st.number_input('Input Screen Size (6.0 - 18.0 Inches)', min_value=6.0, max_value=18.0)

if screen_size:
    st.write(f'You entered a screen size of {screen_size} inches')
else:
    st.write('Please enter a screen size to proceed')


# screen_size = st.number_input('Input Screen Size')

# select resolution of laptop
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

## select cpu

cpu = st.selectbox('CPU', data['cpu_name'].unique())

# select HDD

hdd = st.selectbox('HDD(GB)', [0, 128, 256, 512, 1024, 2048])

## SELECT SSD

ssd = st.selectbox('SSD(GB)', [0, 128, 256, 512, 1024])

# select gpu

gpu = st.selectbox('GPU', data['gpu_brand'].unique())

## conversions

if st.button('Predict Price'):
    
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
        
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
        
    x_resolution = int(resolution.split('x')[0])
    y_resolution = int(resolution.split('x')[1])
    
    ppi = ((x_resolution**2) + (y_resolution**2))**0.5/(screen_size)
    
    query = np.array([company, type, ram, weight, touchscreen,
                      ips, ppi, cpu, hdd, ssd, gpu, os ])
    
    query = query.reshape(1, 12)
    
    # prediction exponent it and convert to integer
    
    prediction = int(np.exp(rf.predict(query)[0]))
    
    # Format the prediction value with commas
    formatted_prediction = f"{prediction:,.0f}"

    # Calculate the range with a 10% deviation
    lower_bound = int(prediction * (1 - 0.1))
    upper_bound = int(prediction * (1 + 0.1))
    
    # Display the result using Streamlit
    st.title(f"Predicted price for this laptop could be between {lower_bound:,} Naira to {upper_bound:,} Naira")

    
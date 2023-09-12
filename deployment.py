import streamlit as st
import pandas as pd
import pickle
import numpy as np

filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

import seaborn as sns
from sklearn import preprocessing
import requests
from PIL import Image
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import joblib

filename='/content/final_model.sav'
loaded_model=pickle.load(open(filename, 'rb'))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Customer Segmentation App")
data=pd.read_csv('/content/Mall_Customers.csv')
st.text('Please enter your data...')

def prepare_input_data_for_model(age, gender, annual_income, spending_score):
    if gender == 'M':
        sex = 1
    else:
        sex = 0

    # Prepare input data as a list
    input_data = [Age, gender, Annual_income, Spending_score]

    # Convert the list to a 2D array
    sample = np.array(input_data).reshape(-1, len(input_data))
    return sample

cluster_names = {
    0: "0-Low annual income, Low spending score",
    1: "1-High annual income, Low spending score",
    2: "2-Low annual income, High spending score",
    3: "3-Moderate Annual Income, Moderate Spending Score",
    4: "4-High Annual Income, High Spending Score",
}

with st.form(key="form1"):
        Age = st.number_input("Age", step=1)
        gender = st.radio('Gender : ', ['F', 'M'])
        Annual_income = st.text_input(label="Annual Income (k$)")
        Spending_score = st.text_input(label="Spending Score (1-100)")
        submit_button=st.form_submit_button(label="Submit")

cluster_df1 = pd.read_csv("/content/Clustered_customer_Data.csv")
if submit_button:
    # Prepare the input data for the model prediction
    input_data = [[Age, 1 if gender == 'M' else 0, Annual_income, Spending_score]]
    
    # Predict the cluster label
    prediction = loaded_model.predict(input_data)
    
    # Display the prediction result with cluster name
    predicted_cluster = prediction[0]  # Assuming prediction is an array with one element
    cluster_name = cluster_names.get(predicted_cluster, "Unknown Cluster")
    
    st.write("Predicted Cluster:")
    st.write(cluster_name)

    # Plot histograms
    plt.rcParams["figure.figsize"] = (20, 3)
    for c in cluster_df1.drop(['Clusters'], axis=1):
      fig, ax = plt.subplots()
      sns.histplot(data=cluster_df1, x=c, hue='Clusters', multiple='stack', ax=ax)
      plt.show()  # Display the plot in a standalone window
      st.pyplot(fig)  # Display the plot in Streamlit
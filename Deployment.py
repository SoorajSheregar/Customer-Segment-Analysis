import streamlit as st
import joblib
import numpy as np

# Load the trained KNN model
model = joblib.load('C:/Users/Sooraj/Downloads/best_knn_model.pkl')  

# Load the scaler object
scaler = joblib.load('C:/Users/Sooraj/Downloads/scaler.pkl')  

# Define the Streamlit app
def main():
    # Set title
    st.title('Customer Segment Prediction')

    # Define input widgets for numerical variables
    income = st.number_input('Income', min_value=10000, max_value=200000, value=50000)
    expenses = st.number_input('Expenses', min_value=0, max_value=10000, value=1000)
    purchases = st.slider('Purchases', min_value=0, max_value=100, value=10)
    age = st.slider('Age', min_value=18, max_value=100, value=30)
    customer_for = st.slider('Customer For (Days)', min_value=0, max_value=700, value=180)
    children = st.slider('Number of Children', min_value=0, max_value=3, value=0)
    accepted_cmp = st.slider('Number of Accepted Campaigns', min_value=0, max_value=5, value=0)
   
    
    # Define input widgets for categorical variables
    education = st.selectbox('Education', ['Under Graduate', 'Post Graduate','PhD'])  # Update with your categories
    living_with = st.selectbox('Living With', ['Alone', 'Partner'])  # Update with your categories
    family_size = st.slider('Family Size', min_value=1, max_value=5, value=2)
    is_parent = st.selectbox('Is Parent', ['No', 'Yes'])  # Update with your categories
    
    # Define a submit button
    submit_button = st.button('Submit')

    # Make prediction based on user inputs when submit button is clicked
    if submit_button:
        # Scale numerical inputs
        scaled_inputs = scaler.transform([[income, children, expenses, accepted_cmp, purchases, age, customer_for]])
        
        # Convert categorical input to numerical labels
        education_label = {'Under Graduate': 0, 'Post Graduate': 1, 'PhD': 2}[education]
        is_parent_label = {'No': 0, 'Yes': 1}[is_parent]
        living_with_label = {'Alone': 0, 'Partner': 1}[living_with]
        
        # Generate predictions
        prediction = model.predict(np.hstack((scaled_inputs, [[education_label, living_with_label, family_size, is_parent_label]])))
        
        # Display prediction
        st.subheader('Prediction')
        st.write(f'The predicted customer segment is: {prediction}')

# Run the Streamlit app
if __name__ == '__main__':
    main()

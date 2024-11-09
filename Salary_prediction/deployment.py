# Import necessary libraries
import streamlit as st

# Import functions from other files
from data_prep import load_data
from feature_eng import preprocess_and_train_model
from modeling import predict_salary
from eda import visualize_data_univariate, visualize_data_bivariate

# Background image CSS
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url('https://martech.org/wp-content/uploads/2022/10/salary.jpg');
background-size: cover; /* Changed to cover */
background-position: center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

# Cache image data
st.markdown(page_bg_img, unsafe_allow_html=True)
# Streamlit UI
def main():
    st.title('SALARY PREDICTION')  

    # Load data
    data = load_data()  

    # Preprocess and train the model
    model, data = preprocess_and_train_model(data)  

    # Sidebar for user input
    st.sidebar.header('Options')  
    option = st.sidebar.radio('Choose an option', ['Prediction', 'Visualization', 'View Dataset'])  

    if option == 'Prediction':  
        st.sidebar.subheader('Prediction Input')  
        company_name = st.sidebar.selectbox('Company Name', data['Company Name'].unique())  
        location = st.sidebar.selectbox('Location', data['Location'].unique())  
        employment_status = st.sidebar.selectbox('Employment Status', data['Employment Status'].unique())  
        job_roles = st.sidebar.selectbox('Job Roles', data['Job Roles'].unique())  
        rating = st.sidebar.number_input('Rating', min_value=0.0, max_value=5.0, step=0.1)  

        if st.sidebar.button('Predict'):  
            predicted_salary = predict_salary(model, company_name, location, employment_status, job_roles, rating)  
            st.success(f'Predicted Salary: {predicted_salary:.2f}')  

    elif option == 'Visualization':  
        st.sidebar.subheader('Visualization Options')  
        visualization_level = st.sidebar.radio('Choose a Visualization Level', ['Univariate', 'Bivariate'])  
        if visualization_level == 'Univariate':  
            univariate_visual = st.sidebar.radio('Choose an Univariate Visualization', ['Bar Chart', 'Pie Chart'])  
            selected_feature_uni = st.sidebar.selectbox('Select Feature', data.columns)  

            if st.sidebar.button('Visualize'):  
                visualize_data_univariate(data, univariate_visual, selected_feature_uni)  
        elif visualization_level == 'Bivariate':  
            bivariate_visual = st.sidebar.radio('Choose a Bivariate Visualization', ['Violin Plot', 'Scatter Plot'])  
            x_feature = st.sidebar.selectbox('Select X Feature', data.columns)  
            y_feature = st.sidebar.selectbox('Select Y Feature', data.columns)  

            if st.sidebar.button('Visualize'):  
                visualize_data_bivariate(data, bivariate_visual, x_feature, y_feature)  

    elif option == 'View Dataset':  
        st.sidebar.subheader('View Dataset Options')  
        view_option = st.sidebar.radio('View Options', ['Specific Feature', 'Full Dataset'])  

        if view_option == 'Specific Feature':  
            feature_to_display = st.sidebar.selectbox('Select Feature', data.columns)  
            st.subheader(f'Displaying Specific Feature: {feature_to_display}')  
            st.write(data[feature_to_display])  
        elif view_option == 'Full Dataset':  
            st.subheader('Full Dataset')  
            st.write(data)  

if __name__ == "__main__":
    main()

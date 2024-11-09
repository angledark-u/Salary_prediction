import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib's pyplot module
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url(https://images.pexels.com/photos/459271/pexels-photo-459271.jpeg?auto=compress&cs=tinysrgb&w=600);
background-size: cover;
background-position: center;
background-repeat: repeat;
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
st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\Vasanth\\Downloads\\ds_salaries.csv")

data = load_data()

# Preprocessing
X = data.drop(columns=['salary', 'salary_currency', 'salary_in_usd'])
y = data['salary_in_usd']

# Define categorical and numerical columns
cat_cols = ['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
num_cols = ['remote_ratio']

# Preprocessing pipeline
num_transformer = SimpleImputer(strategy='median')
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

# Predict function
def predict_salary(work_year, experience_level, employment_type, job_title, employee_residence, remote_ratio, company_location, company_size):
    input_data = pd.DataFrame({
        'work_year': [work_year],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'job_title': [job_title],
        'employee_residence': [employee_residence],
        'remote_ratio': [remote_ratio],
        'company_location': [company_location],
        'company_size': [company_size]
    })
    return model.predict(input_data)[0]

# Streamlit UI
st.title('Salary Prediction ')

# Sidebar buttons
selected_button = st.sidebar.radio('Choose an option:', ('Home Page', 'Salary Prediction', 'Visualization','EDA', 'View Dataset'))

if selected_button == 'Home Page':
    st.markdown("""
    # Predicting Employee Salary with Multiple Linear Regression

    ## Objective
    Our app aims to predict employee salaries using multiple linear regression. We focus on factors such as experience, education level, job role, and more to provide accurate salary estimates for job candidates and help optimize compensation strategies for companies.

    ## Desired Outcomes
    - Accurately predict the salary for a job candidate based on their qualifications and experience.
    - Identify factors that significantly impact employee salaries.
    - Provide valuable insights for HR departments and hiring managers.

  
    """)

if selected_button == 'Salary Prediction':
    st.sidebar.subheader('Input Details')
    work_year = st.sidebar.selectbox('Work Year', data['work_year'].unique())
    experience_level = st.sidebar.selectbox('Experience Level', data['experience_level'].unique())
    employment_type = st.sidebar.selectbox('Employment Type', data['employment_type'].unique())
    job_title = st.sidebar.selectbox('Job Title', data['job_title'].unique())
    employee_residence = st.sidebar.selectbox('Employee Residence', data['employee_residence'].unique())
    remote_ratio = st.sidebar.number_input('Remote Ratio', min_value=0, max_value=100, value=0)
    company_location = st.sidebar.selectbox('Company Location', data['company_location'].unique())
    company_size = st.sidebar.selectbox('Company Size', data['company_size'].unique())

    predict_button = st.sidebar.button('Predict')

    # Prediction result section
    if predict_button:
        predicted_salary = predict_salary(work_year, experience_level, employment_type, job_title, employee_residence, remote_ratio, company_location, company_size)
        st.subheader('Prediction Result')
        st.success(f'Predicted Salary: {predicted_salary:.2f} USD')

elif selected_button == 'Visualization':
    st.title('Dataset Visualization')
    visualization_option = st.sidebar.radio('Choose a visualization:', ('Bar Chart', 'Pie Chart', 'Scatter Plot'))

    if visualization_option == 'Bar Chart':
        st.subheader('Bar Chart')
        bar_column = st.selectbox('Select a column for the bar chart:', data.columns)
        bar_chart_data = data[bar_column].value_counts()
        st.bar_chart(bar_chart_data)

    elif visualization_option == 'Pie Chart':
        st.subheader('Pie Chart')
        pie_column = st.selectbox('Select a column for the pie chart:', data.columns)
        pie_chart_data = data[pie_column].value_counts()
        st.write(pie_chart_data)
        fig, ax = plt.subplots()
        ax.pie(pie_chart_data, labels=pie_chart_data.index, autopct='%1.1f%%')
        st.pyplot(fig)

    elif visualization_option == 'Scatter Plot':
        st.subheader('Scatter Plot')
        scatter_x = st.selectbox('Select X-axis data:', data.columns)
        scatter_y = st.selectbox('Select Y-axis data:', data.columns)
        st.write(f"Scatter plot between {scatter_x} and {scatter_y}")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=scatter_x, y=scatter_y, data=data)
        st.pyplot()
        
elif selected_button == 'EDA':
    st.title('Exploratory Data Analysis')
    
    # Preprocessing - Select numerical columns for correlation analysis
    numeric_data = data.select_dtypes(include=['number'])
    corr_matrix = numeric_data.corr()

    # Display correlation matrix as a heatmap
    st.write('Correlation Matrix:')
    st.write(corr_matrix)
    
    # Save the heatmap plot as an image
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.xlabel('Features')
    plt.ylabel('Features')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')  # Save the figure directly using savefig
    heatmap_image = buffer.getvalue()

    # Display heatmap image
    st.image(heatmap_image, caption='Correlation Matrix Heatmap', use_column_width=True)

    # Explore correlations with salary
    st.subheader('Correlations with Salary')
    salary_corr = corr_matrix['salary_in_usd'].sort_values(ascending=False)
    st.write(salary_corr)

    # Visualize top correlated features with salary
    st.subheader('Top Correlated Features with Salary')
    top_corr_features = salary_corr[1:6]  # Exclude salary itself
    st.write(top_corr_features)
    st.pyplot(sns.barplot(x=top_corr_features.values, y=top_corr_features.index, palette='viridis'))

elif selected_button == 'View Dataset':
    st.title('View Dataset')
    st.write(data)

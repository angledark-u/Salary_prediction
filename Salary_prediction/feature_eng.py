import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Function to preprocess and train the model
def preprocess_and_train_model(data):
    # Preprocessing
    X = data.drop(columns=['Salary', 'Salaries Reported'])  
    y = data['Salary']  

    # Define categorical and numerical columns
    cat_cols = ['Company Name', 'Location', 'Employment Status', 'Job Roles']
    num_cols = ['Rating']

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
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),  
        ('regressor', LinearRegression())  
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)  

    return model, data

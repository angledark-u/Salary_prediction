import pandas as pd

# Function to predict salary based on user input
def predict_salary(model, company_name, location, employment_status, job_roles, rating):
    input_data = pd.DataFrame({
        'Company Name': [company_name],
        'Location': [location],
        'Employment Status': [employment_status],
        'Job Roles': [job_roles],
        'Rating': [rating]
    })
    predicted_salary = model.predict(input_data)[0]  
    return max(predicted_salary, 0)

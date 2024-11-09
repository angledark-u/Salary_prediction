import base64
import pandas as pd

# Function to read image file as base64
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to load data
def load_data():
    # Load the dataset
    data = pd.read_csv("C:\\Users\\judef\\mlpro\\Salary_Dataset_with_Extra_Features.csv")
    return data  # Return loaded dataset

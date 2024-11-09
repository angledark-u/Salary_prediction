import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Function to visualize data univariately
def visualize_data_univariate(data, visualization_type, selected_feature):
    if visualization_type == 'Bar Chart':
        st.subheader('Bar Chart Visualization')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=selected_feature, data=data, ax=ax)
        st.pyplot(fig)
        st.write("Bar charts provide a visual representation of the distribution of categorical data.")
    elif visualization_type == 'Pie Chart':
        st.subheader('Pie Chart Visualization')
        fig, ax = plt.subplots(figsize=(8, 8))
        data[selected_feature].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
        st.write("Pie charts display the proportion of each category in a categorical variable.")

# Function to visualize data bivariately
def visualize_data_bivariate(data, visualization_type, x_feature, y_feature):
    if visualization_type == 'Violin Plot':
        st.subheader('Violin Plot Visualization')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x=x_feature, y=y_feature, data=data, ax=ax)
        st.pyplot(fig)
        st.write("Violin plots are useful for visualizing the distribution of numerical data across different categories.")
    elif visualization_type == 'Scatter Plot':
        st.subheader('Scatter Plot Visualization')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_feature, y=y_feature, data=data, ax=ax)
        st.pyplot(fig)
        st.write("Scatter plots show the relationship between two numerical variables.")

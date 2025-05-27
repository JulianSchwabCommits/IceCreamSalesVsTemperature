import kagglehub
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# Download dataset
path = kagglehub.dataset_download("raphaelmanayon/temperature-and-ice-cream-sales")
print("Path to dataset files:", path)

# Read the CSV file
csv_path = os.path.join(path, "Ice Cream Sales - temperatures.csv")
try:
    df = pd.read_csv(csv_path)
    print("Successfully loaded the dataset")
except FileNotFoundError:
    print(f"Error: Could not find the CSV file at {csv_path}")
    print("Available files in the directory:")
    for file in os.listdir(path):
        print(f"- {file}")
    raise

# Convert temperature from Fahrenheit to Celsius
df['Temperature'] = df['Temperature'].apply(fahrenheit_to_celsius)

# Prepare data for training
X = df[['Temperature']]
y = df['Ice Cream Profits']  # Updated column name

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('ice_cream_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
print(f"Model coefficients: {model.coef_[0]:.2f}")
print(f"Model intercept: {model.intercept_:.2f}")

# Print some statistics about the data
print("\nDataset Statistics:")
print(f"Number of samples: {len(df)}")
print(f"Temperature range: {df['Temperature'].min():.1f}°C to {df['Temperature'].max():.1f}°C")
print(f"Profits range: ${df['Ice Cream Profits'].min():.2f} to ${df['Ice Cream Profits'].max():.2f}")

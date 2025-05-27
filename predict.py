import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import kagglehub
import os

def load_model():
    with open('ice_cream_model.pkl', 'rb') as file:
        return pickle.load(file)

def load_original_data():
    # Download and load the original dataset
    path = kagglehub.dataset_download("raphaelmanayon/temperature-and-ice-cream-sales")
    csv_path = os.path.join(path, "Ice Cream Sales - temperatures.csv")
    df = pd.read_csv(csv_path)
    # Convert temperature to Celsius
    df['Temperature'] = (df['Temperature'] - 32) * 5/9
    return df

def predict_sales(temperature_celsius):
    model = load_model()
    # Create a DataFrame with the correct feature name
    X = pd.DataFrame({'Temperature': [temperature_celsius]})
    prediction = model.predict(X)[0]
    return prediction

def plot_prediction(temperature_celsius, predicted_sales):
    # Load original data
    df = load_original_data()
    
    # Create figure with custom style
    plt.style.use('ggplot')
    
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    
    # Main plot
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('Ice Cream Profits Prediction', fontsize=16, pad=20)
    
    # Create temperature range for smooth curve
    temp_range = np.linspace(0, 40, 100)
    model = load_model()
    # Create DataFrame with proper feature name
    X_range = pd.DataFrame({'Temperature': temp_range})
    sales_range = model.predict(X_range)
    
    # Plot the original data points
    ax1.scatter(df['Temperature'], df['Ice Cream Profits'], 
                color='blue', alpha=0.5, s=50, label='Historical Data')
    
    # Plot the regression line
    ax1.plot(temp_range, sales_range, 'g-', alpha=0.7, 
             label='Regression Line', linewidth=2)
    
    # Plot the prediction point
    ax1.scatter(temperature_celsius, predicted_sales, color='red', s=200, 
                label=f'Prediction: ${predicted_sales:.2f}', zorder=5)
    
    # Customize the plot
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Ice Cream Profits ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add prediction details
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    prediction_text = f"""
    Temperature: {temperature_celsius:.1f}°C
    Predicted Profits: ${predicted_sales:.2f}
    """
    ax2.text(0.5, 0.5, prediction_text, 
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        temperature = float(input("Enter temperature in Celsius: "))
        predicted_sales = predict_sales(temperature)
        plot_prediction(temperature, predicted_sales)
    except ValueError:
        print("Please enter a valid number for temperature.")
    except FileNotFoundError:
        print("Model file not found. Please run train_model.py first.")

if __name__ == "__main__":
    main()

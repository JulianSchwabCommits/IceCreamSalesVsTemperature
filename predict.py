import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub

def predict_and_plot(temperature_celsius):
    # Load model and data
    model = pickle.load(open('ice_cream_model.pkl', 'rb'))
    df = pd.read_csv(kagglehub.dataset_download("raphaelmanayon/temperature-and-ice-cream-sales") + "/Ice Cream Sales - temperatures.csv")
    df['Temperature'] = (df['Temperature'] - 32) * 5/9
    
    # Make prediction
    predicted_sales = model.predict(pd.DataFrame({'Temperature': [temperature_celsius]}))[0]

    min_temp = min(0, df['Temperature'].min(), temperature_celsius)
    max_temp = max(40, df['Temperature'].max(), temperature_celsius)
    temp_range = np.linspace(min_temp, max_temp, 200)
    sales_range = model.predict(pd.DataFrame({'Temperature': temp_range}))
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Temperature'], df['Ice Cream Profits'], color='blue', alpha=0.5, label='Historical Data')
    plt.plot(temp_range, sales_range, 'g-', label='Regression Line')
    plt.scatter(temperature_celsius, predicted_sales, color='red', s=100, label=f'Prediction: ${predicted_sales:.2f}')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Ice Cream Profits ($)')
    plt.title('Ice Cream Profits Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(min_temp, max_temp)
    plt.show()

if __name__ == "__main__":
    try:
        temperature = float(input("Enter temperature in Celsius: "))
        predict_and_plot(temperature)
    except ValueError:
        print("Please enter a valid number for temperature.")
    except FileNotFoundError:
        print("Model file not found. Please run train_model.py first.")

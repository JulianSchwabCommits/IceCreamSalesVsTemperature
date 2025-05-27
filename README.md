# Ice Cream Sales vs Temperature Prediction Model

A machine learning project that predicts ice cream profits based on temperature using linear regression. The model is trained on historical data and can make predictions for any given temperature in Celsius.

## Dataset
The project uses a dataset from Kaggle that contains historical data of ice cream sales and corresponding temperatures:
[Temperature and Ice Cream Sales Dataset](https://www.kaggle.com/datasets/raphaelmanayon/temperature-and-ice-cream-sales)

## Features
- Temperature conversion from Fahrenheit to Celsius
- Linear regression model for profit prediction
- Interactive visualization of predictions
- Historical data visualization
- Beautiful matplotlib plots with custom styling

## Dependencies
The project uses the following Python packages:
- `kagglehub`: For downloading the dataset
- `pandas`: For data manipulation and analysis
- `numpy`: For numerical operations
- `scikit-learn`: For the linear regression model
- `matplotlib`: For creating visualizations
- `seaborn`: For enhanced plot styling

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/JulianSchwabCommits/IceCreamSalesVsTemperature.git
cd IceCreamSalesVsTemperature
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python train_model.py
```
This will:
- Download the dataset
- Convert temperatures to Celsius
- Train the linear regression model
- Save the model as 'ice_cream_model.pkl'

2. Make predictions:
```bash
python predict.py
```
This will:
- Load the trained model
- Prompt you to enter a temperature in Celsius
- Display a visualization showing:
  - Historical data points (blue dots)
  - Regression line (green line)
  - Your prediction point (red dot)

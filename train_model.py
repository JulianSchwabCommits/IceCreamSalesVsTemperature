import kagglehub
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Download and load data
df = pd.read_csv(kagglehub.dataset_download("raphaelmanayon/temperature-and-ice-cream-sales") + "/Ice Cream Sales - temperatures.csv")
df['Temperature'] = (df['Temperature'] - 32) * 5/9  # Convert to Celsius

# Train model
model = LinearRegression()
model.fit(df[['Temperature']], df['Ice Cream Profits'])

# Save model
pickle.dump(model, open('ice_cream_model.pkl', 'wb'))

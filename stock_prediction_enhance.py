import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Data Loading
company = input("Enter the company name: ")
path = "D:\\Python_Project\\" + company + ".csv"
print(f"Loading data from: {path}")

# Load the dataset
df = pd.read_csv(path)

# Step 2: Data Preprocessing
# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Drop unnecessary columns like 'Adj Close'
df.drop('Adj Close', axis=1, inplace=True)

# Check for missing values and handle them
df = df.dropna()

# Convert 'Volume' column to float if not already
df['Volume'] = df['Volume'].astype(float)

# Step 3: Feature Selection
# We will use Open, High, Low, Volume to predict the Close price
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Building - Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 8: Visualization - Actual vs Predicted Prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual Price', color='b')
plt.plot(y_pred, label='Predicted Price', color='r')
plt.title(f"Actual vs Predicted Stock Prices for {company}")
plt.xlabel("Time")
plt.ylabel("Stock Closing Price")
plt.legend()
plt.show()

# Step 9: Display Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

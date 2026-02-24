import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

file_path = '/content/MLDATASET.csv'
data = pd.read_csv(file_path, delimiter=',')

expected_columns = ['Year', 'MAKE', 'MODEL', 'VEHICLE CLASS', 'ENGINE SIZE', 'CYLINDERS', 'COMESSION']
if len(data.columns) == len(expected_columns):
    data.columns = expected_columns
else:
    raise ValueError(f"Unexpected number of columns: {len(data.columns)}. Expected {len(expected_columns)}.")

data['VEHICLE CLASS'] = data['VEHICLE CLASS'].astype('category').cat.codes

data['ENGINE SIZE'] = pd.to_numeric(data['ENGINE SIZE'], errors='coerce')
data['CYLINDERS'] = pd.to_numeric(data['CYLINDERS'], errors='coerce')
data['COMESSION'] = pd.to_numeric(data['COMESSION'], errors='coerce')

print(f"\nNumber of rows before dropping missing values: {len(data)}")
data = data.dropna()
print(f"Number of rows after dropping missing values: {len(data)}")

features = ['ENGINE SIZE', 'CYLINDERS', 'VEHICLE CLASS', 'COMESSION']
X = data[features]
y = data['COMESSION'] * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'\nMean Absolute Error: {mae}')

data['Predicted Commission'] = model.predict(X)

print("\nPredicted Commissions for the first few vehicles:")
print(data[['MAKE', 'MODEL', 'Predicted Commission']].head())


all_make_model_predictions = data.groupby(['MAKE', 'MODEL'])['Predicted Commission'].mean().reset_index()
print("\nPredicted CO2 Emissions (Commissions) by Make and Model for All Cars:")
print(all_make_model_predictions)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(data['COMESSION'] * 0.1, kde=True, color='blue', label='Actual Commission')
plt.title('Distribution of Actual Commissions')
plt.xlabel('Commission')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(data['Predicted Commission'], kde=True, color='red', label='Predicted Commission')
plt.title('Distribution of Predicted Commissions')
plt.xlabel('Commission')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data['COMESSION'] * 0.1, data['Predicted Commission'], alpha=0.5, color='green')
plt.plot([min(data['COMESSION'] * 0.1), max(data['COMESSION'] * 0.1)],
         [min(data['COMESSION'] * 0.1), max(data['COMESSION'] * 0.1)],
        color='red', linestyle='--', label='Perfect Prediction')
plt.title('Actual vs Predicted Commissions')
plt.xlabel('Actual Commission')
plt.ylabel('Predicted Commission')
plt.legend()
plt.show()

import numpy as np  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 
tractor_age = np.linspace(5, 25, 52).reshape(-1, 1)

np.random.seed(42)  
maintenance_cost = (-20 * (tractor_age - 15)**2 + 1600 +
                    np.random.normal(0, 100, size=tractor_age.shape[0]))

X_train, X_test, y_train, y_test = train_test_split(tractor_age, maintenance_cost, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False) 
X_train_poly = poly.fit_transform(X_train)  
X_test_poly = poly.transform(X_test)  

model = LinearRegression()  
model.fit(X_train_poly, y_train) 

y_pred = model.predict(X_test_poly) 

mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

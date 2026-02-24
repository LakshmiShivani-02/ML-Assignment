import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)
tractor_age = np.linspace(5, 25, 52).reshape(-1, 1) 
maintenance_cost = (-20 * (tractor_age.flatten() - 15)**2 + 1600 +
                    np.random.normal(0, 100, size=tractor_age.shape[0]))  

poly = PolynomialFeatures(degree=2, include_bias=False)
tractor_age_poly = poly.fit_transform(tractor_age)  
model = LinearRegression()
model.fit(tractor_age_poly, maintenance_cost)  

tractor_age_range = np.linspace(5, 25, 100).reshape(-1, 1) 
tractor_age_range_poly = poly.transform(tractor_age_range)
predicted_costs = model.predict(tractor_age_range_poly)

plt.scatter(tractor_age, maintenance_cost, color='blue', label='Actual Data')  
plt.plot(tractor_age_range, predicted_costs, color='red', label='Predicted Curve')  
plt.title('Tractor Age vs Maintenance Cost')
plt.xlabel('Tractor Age (Years)')
plt.ylabel('Maintenance Cost')
plt.legend()
plt.grid(True)
plt.show()

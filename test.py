import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Generate some example data
x_data = np.linspace(0, 10, 100)
y_data = 2 * np.exp(-0.5 * x_data) 

# Transform the data using the natural logarithm
y_data_transformed = np.log(y_data)
print(y_data_transformed)

# Define the Gaussian function to fit
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# Perform the curve fit on the transformed data
popt, pcov = curve_fit(gaussian, x_data, y_data_transformed, p0=[1, 0, 1])

# Generate the fitted curve over the full range of data
y_fit_all = np.exp(gaussian(x_data, *popt))

# Plot the data and fitted curve
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, y_fit_all, label='Fit')
plt.legend()
plt.show()
plt.savefig("test.png")

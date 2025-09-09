# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("avocado.csv", parse_dates=['Date'])
data = data.groupby('Date')['Total Volume'].sum().to_frame()

resampled_data = data.resample('Y').sum()
resampled_data.index = resampled_data.index.year
 
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Date': 'Year', 'Total Volume': 'TotalVolume'}, inplace=True)

years = resampled_data['Year'].tolist()
volumes = resampled_data['TotalVolume'].tolist()

X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, volumes)]

n = len(years)
b = (n * sum(xy) - sum(volumes) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(volumes) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, volumes)]

coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(volumes), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend

resampled_data.set_index('Year', inplace=True)

resampled_data['TotalVolume'].plot(kind='line', color='blue', marker='o', label='Total Volume')
resampled_data['Linear Trend'].plot(kind='line', color='black', linestyle='--', label='Linear Trend')
plt.legend()
plt.title("Linear Trend Estimation on Avocado Total Volume")
plt.show()

resampled_data['TotalVolume'].plot(kind='line', color='blue', marker='o', label='Total Volume')
resampled_data['Polynomial Trend'].plot(kind='line', color='red', marker='o', label='Polynomial Trend (Degree 2)')
plt.legend()
plt.title("Polynomial Trend Estimation on Avocado Total Volume")
plt.show()
```


### OUTPUT
#### A - LINEAR TREND ESTIMATION
<img width="768" height="620" alt="image" src="https://github.com/user-attachments/assets/4a831e6f-faee-42af-9763-f8c249690433" />


#### B- POLYNOMIAL TREND ESTIMATION
<img width="762" height="593" alt="image" src="https://github.com/user-attachments/assets/ecea9cae-8d5b-4a3f-9db4-02fd3086e92c" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.

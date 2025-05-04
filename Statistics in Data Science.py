# Module 6 - Statistics in Data Science

# 1. Impact of Outliers on Mean, Median, Mode
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import seaborn as sns

# a. Create dataset with outliers
data_with_outliers = [10, 12, 13, 12, 11, 10, 100]  # 100 is an outlier

def summarize_data(data):
    mean = np.mean(data)
    median = np.median(data)
    try:
        mode = stats.mode(data)
    except:
        mode = 'No unique mode'
    return mean, median, mode

# Before removing outliers
mean1, median1, mode1 = summarize_data(data_with_outliers)

# Remove outlier (simple rule: values > 3*std dev)
z_scores = (data_with_outliers - np.mean(data_with_outliers)) / np.std(data_with_outliers)
data_without_outliers = [x for x, z in zip(data_with_outliers, z_scores) if abs(z) < 2]
mean2, median2, mode2 = summarize_data(data_without_outliers)

print("With Outliers → Mean:", mean1, ", Median:", median1, ", Mode:", mode1)
print("Without Outliers → Mean:", mean2, ", Median:", median2, ", Mode:", mode2)

# 2. Measures of Central Tendency with Histogram
test_scores = [45, 50, 55, 60, 65, 65, 70, 75, 80, 85, 85, 90]
mean = stats.mean(test_scores)
median = stats.median(test_scores)
mode = stats.mode(test_scores)

plt.figure(figsize=(8, 5))
sns.histplot(test_scores, bins=8, kde=False)
plt.axvline(mean, color='r', linestyle='--', label='Mean')
plt.axvline(median, color='g', linestyle='--', label='Median')
plt.axvline(mode, color='b', linestyle='--', label='Mode')
plt.title("Histogram with Mean, Median, Mode")
plt.legend()
plt.show()

# 3. Measures of Dispersion
import pandas as pd

data = [55, 60, 65, 70, 75, 80, 85, 90]
range_val = np.max(data) - np.min(data)
q75, q25 = np.percentile(data, [75 ,25])
iqr = q75 - q25
variance = np.var(data)
std_dev = np.std(data)

print("Range:", range_val, "IQR:", iqr, "Variance:", variance, "Std Dev:", std_dev)

# Boxplot
plt.figure()
plt.boxplot(data, vert=False)
plt.title("Boxplot of Data")
plt.show()

# 4. Paired t-test
from scipy.stats import ttest_rel

before = [65, 70, 60, 62, 66, 64, 63]
after = [68, 75, 65, 70, 70, 69, 68]
t_stat, p_val = ttest_rel(after, before)
print("Paired t-test → t-stat:", t_stat, ", p-value:", p_val)
if p_val < 0.05:
    print("Result: Significant improvement")
else:
    print("Result: No significant change")

# 5. Type I and Type II Error Simulation
from scipy.stats import norm

def simulate_errors():
    # Type I Error: Rejecting true null
    data1 = norm.rvs(loc=0, scale=1, size=30)  # Null is true
    result1 = ttest_rel(data1, norm.rvs(loc=0, scale=1, size=30))

    # Type II Error: Failing to reject false null
    data2_before = norm.rvs(loc=0, scale=1, size=30)
    data2_after = norm.rvs(loc=0.5, scale=1, size=30)  # Slight improvement
    result2 = ttest_rel(data2_before, data2_after)

    print("Type I Error Scenario → p-value:", result1.pvalue)
    print("Type II Error Scenario → p-value:", result2.pvalue)

simulate_errors()

# 6. Simple Linear Regression (Experience vs Salary)
from sklearn.linear_model import LinearRegression

experience = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
salary = np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000])

model = LinearRegression()
model.fit(experience, salary)
predicted_salary = model.predict([[5]])
print("Predicted salary for 5 years experience:", predicted_salary[0])

# Scatter and regression line
plt.figure()
plt.scatter(experience, salary, color='blue')
plt.plot(experience, model.predict(experience), color='red')
plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# 7. Underfitting vs Overfitting
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate data
def generate_data():
    X = np.linspace(-3, 3, 100)
    y = X**2 + np.random.normal(0, 1, size=X.shape)
    return X.reshape(-1, 1), y

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Linear model (underfitting)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# High-degree poly model (overfitting)
poly = PolynomialFeatures(degree=15)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Predictions and MSE
y_pred_lin = lin_model.predict(X_test)
y_pred_poly = poly_model.predict(X_poly_test)
print("Linear Model MSE:", mean_squared_error(y_test, y_pred_lin))
print("Polynomial Model MSE:", mean_squared_error(y_test, y_pred_poly))

# Visualization
plt.figure()
plt.scatter(X_test, y_test, color='blue', label='True')
plt.plot(X_test, y_pred_lin, color='green', label='Linear')
plt.plot(X_test, y_pred_poly, color='red', label='Polynomial')
plt.title("Underfitting vs Overfitting")
plt.legend()
plt.show()

# 8. Compare Regression Types
from sklearn.linear_model import Lasso

X = np.linspace(1, 10, 50)
y = 3 * X + np.random.randn(50) * 3
X = X.reshape(-1, 1)

# Linear
model_linear = LinearRegression().fit(X, y)
mse_linear = mean_squared_error(y, model_linear.predict(X))

# Polynomial
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression().fit(X_poly, y)
mse_poly = mean_squared_error(y, model_poly.predict(X_poly))

# Lasso
model_lasso = Lasso(alpha=0.1).fit(X, y)
mse_lasso = mean_squared_error(y, model_lasso.predict(X))

print("MSE → Linear:", mse_linear, ", Polynomial:", mse_poly, ", Lasso:", mse_lasso)

# Visualization
plt.figure()
plt.scatter(X, y, color='gray', label='Data')
plt.plot(X, model_linear.predict(X), label='Linear')
plt.plot(X, model_poly.predict(X_poly), label='Polynomial')
plt.plot(X, model_lasso.predict(X), label='Lasso')
plt.title("Regression Comparisons")
plt.legend()
plt.show()

# 9. Correlation and Regression
import numpy as np

study_hours = np.array([1, 2, 3, 4, 5, 6, 7])
scores = np.array([50, 55, 60, 65, 70, 75, 80])

correlation = np.corrcoef(study_hours, scores)[0, 1]
print("Pearson Correlation:", correlation)

model = LinearRegression()
model.fit(study_hours.reshape(-1, 1), scores)

# Scatter with regression line
plt.figure()
plt.scatter(study_hours, scores, color='blue')
plt.plot(study_hours, model.predict(study_hours.reshape(-1, 1)), color='red')
plt.title("Study Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

#  READ CSV 
df = pd.read_csv("/Users/mac/Documents/GITHUB REPOSITORY/Section-A-Q1-USA-Real-Estate-Dataset-realtor-data-Rev1.csv")

#  DATA reading
print("Datatypes of CSV")
print(df.dtypes)

print("Information of CSV")
print(df.info())

print("Statistics of CSV")
print(df.describe())

print("Shape of DataFrame")
print(df.shape)

print("First 10 rows")
print(df.head(10))

print("Last 10 rows")
print(df.tail(10))

# COLUMN & ROW SELECTION
print("Single column (city)")
print(df['city'])

print("Multiple columns (bed, bath)")
print(df[['bed', 'bath']])

print("Single row")
print(df.loc[1])

print("Multiple rows")
print(df.loc[[1, 3]])

# deleting ROW 
df.drop(index=2, inplace=True, errors="ignore")
df.drop(index=[4, 5], inplace=True, errors="ignore")

#  RENAME COLUMN 
df.rename(columns={'status': 'status_of_property'}, inplace=True)

# NUMPY OPERATIONS 
brokered_by, price, acre_lot = np.genfromtxt("/Users/mac/Documents/USA_REALESTATE/Section-A-Q1-USA-Real-Estate-Dataset-realtor-data-Rev1.csv",delimiter=",",usecols=(0, 2, 5),skip_header=1,unpack=True)

print("Mean Acre:", np.nanmean(acre_lot))
print("Std Acre:", np.nanstd(acre_lot))
print("Median Acre:", np.nanmedian(acre_lot))
print("Min Acre:", np.nanmin(acre_lot))
print("Max Acre:", np.nanmax(acre_lot))
print("Mean Price:", np.nanmean(price))

#  MATH OPERATIONS
print("Square brokered_by:", np.square(brokered_by))
print("Sqrt brokered_by:", np.sqrt(np.abs(brokered_by)))
print("Absolute price:", np.abs(price))

# ARITHMETIC OPERATIONS
print("Addition:", brokered_by + price)
print("Subtraction:", brokered_by - price)
print("Multiplication:", brokered_by * price)
print("Division:", brokered_by / price)

#  VISUALIZATION 
variables = ["brokered_by", "bed", "bath", "acre_lot"]

for var in variables:
    plt.figure()
    sns.regplot(x=var, y="price", data=df, scatter_kws={"alpha": 0.3})
    plt.title(f"Regression plot of {var} vs Price")
    plt.show()

# MACHINE LEARNING

# Remove rows where (price) is NaN
df = df.dropna(subset=["price"])

# Independent & dependent variables
X = df[["brokered_by", "bed", "bath", "acre_lot", "zip_code", "house_size"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

# using imputer due to nan values as they cant be used in linear regression
#  Imputer + Scaler + Linear Regression
pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler()),("model", LinearRegression())])
pipeline.fit(X_train, y_train)

# Cross-validation
mse = cross_val_score(pipeline,X_train,y_train,scoring="neg_mean_squared_error",cv=6)

print("Mean MSE:", np.mean(mse))

# Prediction
reg_pred = pipeline.predict(X_test)

# aqqurecy graph between real and predicted value by machine
sns.displot(reg_pred - y_test, kind="kde")
plt.title("Prediction Error Distribution")
plt.show()

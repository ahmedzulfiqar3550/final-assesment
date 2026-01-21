import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("/Users/mac/Documents/GITHUB REPOSITORY/housing[1].csv")

print(df.head())
print(df.tail())
print(df.shape)
print(df.describe())

variables=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']
for var in variables:
    plt.figure() 
    sns.regplot(x=var, y='median_house_value', data=df).set(title=f'Regression plot of {var} and median_housing_value');
    plt.show()

y = df['median_housing_value']
X = df[['longitude,latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=200)

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)
print(y_pred)   

from sklearn.metrics import mean_absolute_error,mean_squared_error

erro1 = mean_absolute_error(y_test, y_pred)
erro2 = mean_squared_error(y_test, y_pred)
erro3= np.sqrt(erro2)

print(f'Mean absolute error: {erro1:.2f}')
print(f'Mean squared error: {erro2:.2f}')
print(f'Root mean squared error: {erro3:.2f}')
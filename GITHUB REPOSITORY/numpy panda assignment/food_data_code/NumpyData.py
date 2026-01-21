import numpy as np

latitude, postalcode= np.genfromtxt("/Users/mac/Documents/GITHUB REPOSITORY/numpy panda assignment/food_data_code/FastFoodRestaurants.csv",delimiter=",", usecols=(4,7), dtype=None, skip_header=1, unpack=True,invalid_raise=False)
print(postalcode)
print(latitude)

# performing statistics operations
print("the mean value of latitude=",np.mean(latitude))
print("the standard devision of latitude=",np.std(latitude))
print("the average of latitude =",np.average(latitude))
print("the median of latitude =",np.median(latitude))
print("the minimum latitude =",np.min(latitude))
print("the maximum latitude =",np.max(latitude))

# performing math operations
print("the square of latitude=",np.square(latitude))
print("the square root of latitude=",np.sqrt(latitude))
print("the power of latitude=",np.power(latitude,latitude))
print("the absolute value of latitude=",np.abs(latitude))

#performing arethmatic operations 
addition = latitude + postalcode
subtraction = latitude - postalcode
multiplication = latitude * postalcode
devision = latitude/postalcode

print("addition =",addition)
print("subtraction =",subtraction)
print("multiplication =",multiplication)
print("devision =",devision)


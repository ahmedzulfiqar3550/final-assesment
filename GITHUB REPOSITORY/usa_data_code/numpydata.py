import numpy as np
price , bath = np.genfromtxt("/Users/mac/Documents/GITHUB REPOSITORY/usa_data_code/RealEstate-USA.csv",delimiter=",", usecols=(2,4), dtype=float, skip_header=1, unpack=True)

print(price)
print(bath)

# performing statistics operations
print("the mean value of price=",np.mean(price))
print("the standard devision of price=",np.std(price))
print("the average of price =",np.average(price))
print("the median of price =",np.median(price))
print("the minimum price =",np.min(price))
print("the maximum price =",np.max(price))

# performing math operations
print("the square of price=",np.square(price))
print("the square root of price=",np.sqrt(price))
print("the power of price=",np.power(price,price))
print("the absolute value of price=",np.abs(price))

#performing arethmatic operations 
addition = price + bath
subtraction = price - bath
multiplication = price * bath
devision = price/bath

print("addition =",addition)
print("subtraction =",subtraction)
print("multiplication =",multiplication)
print("devision =",devision)


import pandas as pd

df = pd.read_csv("/Users/mac/Documents/GITHUB REPOSITORY/numpy panda assignment/food_data_code/FastFoodRestaurants.csv",delimiter=",")
print(df) 

print("information=",df.info)

print("last four row=",df.tail(4))
print("first four rows=",df.head(4))

print("statistics using describe()=",df.describe)
print("counting the rows and columns using shape=",df.shape)

city=df["city"]
print("accessing the city coulmn=",city)

print("accessing multiple columns=")
city_name=df[['city','name']]
print(city_name)

# printing rows and columns using loc and iloc
row=df.loc[0:3]
print(row)

coln=df.loc[:,"latitude"]
print(coln)

coln_2=df.loc[:5,"longitude":"name"]
print(coln_2)

col = df.iloc[:,0:7]
print(col)

cas = df.iloc[[1,3],0:9]
print(cas)


df.drop(2, axis=0, inplace=True)
print(df)

df.drop("city", axis=1, inplace=True)
print(df)

df.rename(columns={"country" : "latitude"}, inplace=True)
print(df)
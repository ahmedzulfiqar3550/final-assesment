import pandas as pd

df = pd.read_csv("/Users/mac/Documents/GITHUB REPOSITORY/usa_data_code/RealEstate-USA.csv",delimiter="," )
print(df)

print(df.shape)

print('Last three Rows:')
print(df.tail(3))

print('First Three Rows:')
print(df.head(3))

row = df.loc[0:4]
print(row)

col = df.loc[:,"bed"]
print(col)

col = df.iloc[:,0:7]
print(col)

cas = df.iloc[[1,3],0:9]
print(cas)


df.drop(2, axis=0, inplace=True)
print(df)

df.drop("price", axis=1, inplace=True)
print(df)

df.rename(columns={"bed" : "bedroom"}, inplace=True)
print(df)


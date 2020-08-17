import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
import plotly.express as px
df= pd.read_csv('house_prices_test.csv')
#print(df.head())
#print(df)
#print(df.columns)
#print(df.shape)

lat = []
lon = []
for row in df['Location']:
    try:
        lat.append(row.split(',')[0])
        lon.append(row.split(',')[1])
    except:
        lat.append(np.NaN)
        lon.append(np.NaN)

df['latitude'] = lat
df['longitude'] = lon
# print(df.columns)
# print(df.head())

for column in df.columns:
    df['latitude'] = df['latitude'].apply(lambda x: re.sub("[a-z]+:", " ", x))
# print(df.head(2))
# print(df.columns)
for columns in df.columns:
    df['longitude'] = df['longitude'].apply(lambda x: re.sub("[a-z]+:", " ",x ))
# print(df.head(2))
df['city'] = 1
df = df.drop(columns = "ID")
print(df.head(10))
df['year'] = pd.DatetimeIndex(df['Listing Date']).year
print(df['year'])
df['Current_Year'] = 2020
df['no_year'] = df['Current_Year'] - df['year']

df.drop(["latitude", "longitude", "Current_Year", "Listing Date", "year" ], axis = 1, inplace = True)
#df = pd.get_dummies(df, drop_first=True)
print(df.columns)
print(df.head(10))
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'Missing catogory': df.columns,'percent_missing': percent_missing})
print(missing_value_df)
#Converting to integer and putting in new column "Size in meters"
size = []
#for row in df['Erf Size']:
 #   try:
  #      size.append(row.split(' ')[0])
   # except:
    #    size.append(np.NaN)
#df['Size_meters_Square'] = size
#for column in df.columns:
#    df['Size_meters_Square'] = df['Size_meters_Square'].astype('str')
#    df['Size_meters_Square'] = df['Size_meters_Square'].apply(lambda x: re.sub("[a-z]+:", " ", x))
df['Erf Size'] = df['Erf Size'].str.replace(' ', '')
df['Erf Size'] = df['Erf Size'].str[:-2]
df['Erf Size'] = df['Erf Size'].fillna(0)
size = []
for row in df['Erf Size']:
    try:
        size.append(row.split(' ')[0])
    except:
        size.append(np.NaN)
df['Size_meters_Square'] = size
for column in df.columns:
    df['Size_meters_Square'] = df['Size_meters_Square'].astype('str')
    df['Size_meters_Square'] = df['Size_meters_Square'].apply(lambda x: re.sub("[a-z]+:", " ", x))
df['Size_meters_Square'] = df['Erf Size'].fillna(0)
print(df.dtypes)
df['Erf Size'] = pd.to_numeric(df['Erf Size'], errors='coerce')
df = df.dropna(subset=['Erf Size'])
#df['Erf Size']= df['Erf Size'].str.replace(',', '').astype('int')
#df['Erf Size'] = df["Erf Size"]. astype(int)
print("this")
print(df['Erf Size'])
df['Type of Property'] = pd.get_dummies(df['Type of Property'], drop_first=True)
#df['new_col'] = df['Size_meters_Square'].astype(str).str[:-2]
print(df.head())
print(df.dtypes)
df = df.drop(['Location'], axis = 1)
df = df.dropna()
X = df.iloc[:,4:]
y = df.iloc[:,4]
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
df.to_csv(r'C:\Users\hp\PycharmProjects\machine_learning_kaggle\sirosh_new_test.csv', index = False)
corrmat= df.corr()
top_corr_featuers = corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(df[top_corr_featuers].corr(), annot=True,cmap="RdYlGn")
plt.show()

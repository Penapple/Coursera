import pandas as pd
import numpy as np
train_data, test_data = pd.read_csv('kc_house_train_data.csv', dtype={
    'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,\
    'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,\
    'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int,\
    'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int
}), pd.read_csv('kc_house_test_data.csv', dtype={
    'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,\
    'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,\
    'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int,\
    'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int
})

bedrooms_squared = test_data['bedrooms']*test_data['bedrooms']
print(bedrooms_squared.mean())

print((test_data['bedrooms']*test_data['bathrooms']).mean())

print(np.log(test_data['sqft_living']).mean())

print((test_data['lat'] + test_data['long']).mean())

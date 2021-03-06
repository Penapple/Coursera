import pandas as pd
import numpy as np
from sklearn import linear_model
data = pd.read_csv('kc_house_train_data.csv', dtype={
    'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,\
    'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,\
    'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int,\
    'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int
})

bed_bath_rooms = data['bedrooms']*data['bathrooms']
bedrooms_squared = data['bedrooms']*data['bedrooms']
log_sqft_living = np.log(data['sqft_living'])
lat_plus_long = data['lat'] + data['long']

x1 = []
y1 = []
for a,b,c,d,e, output in zip(data['sqft_living'],data['bedrooms'],data['bathrooms'],data['lat'],data['long'], data['price']):
    x1.append([float(a), float(b), float(c), float(d), float(e)])
    y1.append(float(output))
regr = linear_model.LinearRegression()
regr.fit(x1,y1)
print(regr.coef_)
print(np.sum((regr.predict(x1)-y1)**2))

x2 = []
y2 = []
for a,b,c,d,e,f, output in zip(data['sqft_living'],data['bedrooms'],data['bathrooms'],data['lat'],data['long'], bed_bath_rooms, data['price']):
    x2.append([float(a), float(b), float(c), float(d), float(e), float(f)])
    y2.append(float(output))
regr = linear_model.LinearRegression()
regr.fit(x2,y2)
print(regr.coef_)
print(np.sum((regr.predict(x2)-y2)**2))

x3 = []
y3 = []
for a,b,c,d,e,f,g,h,i, output in zip(data['sqft_living'],data['bedrooms'],data['bathrooms'],data['lat'],data['long'], bed_bath_rooms,bedrooms_squared,log_sqft_living, lat_plus_long, data['price']):
    x3.append([float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h), float(i)])
    y3.append(float(output))
regr = linear_model.LinearRegression()
regr.fit(x3,y3)
print(regr.coef_)
print(np.sum((regr.predict(x3)-y3)**2))

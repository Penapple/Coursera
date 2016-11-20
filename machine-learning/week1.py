import pandas as pd
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

def simple_linear_regression(input_feature, output):
    slope = ((input_feature*output).sum() - input_feature.sum() * output.mean())/((input_feature ** 2).sum() - input_feature.sum() * input_feature.mean())
    intercept = output.mean() - slope * input_feature.mean()
    return(intercept, slope)

input_feature = train_data['sqft_living']
output = train_data['price']

print(simple_linear_regression(input_feature, output))

def get_regression_predictions(input_feature, intercept, slope):
    predicted_output = intercept + slope * input_feature
    return(predicted_output)

print(get_regression_predictions(2650, -47116.07907289418, 281.9588396303426))

def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    predicted_values = intercept + (slope * input_feature)
    residuals = output - predicted_values
    RSS = (residuals * residuals).sum()
    return(RSS)

print (get_residual_sum_of_squares(input_feature, output, -47116.07907289418, 281.9588396303426))

def inverse_regression_predictions(output, intercept, slope):
    estimated_input = (output - intercept)/slope
    return(estimated_input)

print(inverse_regression_predictions(800000, -47116.07907289418, 281.9588396303426))

print (get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], -47116.07907289418, 281.9588396303426))


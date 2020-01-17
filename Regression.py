import numpy as np
import pandas as pd
import statsmodels.api as sm


def eliminate_outliers(dataset):
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    dataset = dataset[~((dataset < (Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return (dataset)


dataset = pd.read_csv('akadole3.csv', names = ['X1','X2','X3','X4','X5','Y'])

column = ['X1', 'X2', 'X3', 'X4', 'X5', 'Y']
X = dataset[column]
X = eliminate_outliers(X)
Y = X['Y']  
X = X[column[:-1]]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
prediction = model.predict()
print(prediction)
s = np.sum((prediction - Y)**2)
print("Sum_squared_error for test data is : %0.2f" % s)
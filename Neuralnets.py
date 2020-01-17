import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.optimizers import SGD
#from sklearn.model_selection import GridSearchCV

data = pd.read_csv("akadole1.csv", header=None)

train = data 

X_in = train.iloc[:,0:5]   # Includes all rows but only first 5 columns
Y_out = train.iloc[:,5]    # include all rows and the last column
Y = np.reshape(Y_out.values, (-1,1))     # next steps include normalizing the data so that all variable values are in the same range
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X_in))
xscale = scaler_x.transform(X_in)
print(scaler_y.fit(Y))
yscale = scaler_y.transform(Y)
scale = 2000
X_train = xscale[:scale]        #Dividing the data into training and testing data
Y_train = yscale[:scale]
X_test = xscale[scale:]
Y_test = Y_out[scale:]
Y_test1 = yscale[scale:]

def baseline_model(learn_rate = 0.001, momentum = 0.9):
	# create model
    model = Sequential()
    model.add(Dense(16, input_dim=5, kernel_initializer='normal', activation='relu'))    # single hidden layer with 16 hidden nodes 
    model.add(Dense(1))          # Output with single node
    optimizer = SGD(lr = learn_rate, momentum = momentum)     
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer = optimizer, metrics = ['mae'])
    return model

# evaluate model with standardized dataset
f_model = baseline_model()
history = f_model.fit(X_train, Y_train, validation_data=(X_test, Y_test1), epochs = 50, batch_size = 10)        #training for the fixed size of epochs
model = KerasRegressor(build_fn = baseline_model, verbose=1)

#####################  Automatic Grid Search ########################

#batch_size = [10, 20, 40, 60, 80, 100]
#epochs = [10, 20, 40, 50, 80, 100]
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#neurons = [2, 4, 12, 16, 32, 64]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#param_grid = dict(batch_size = batch_size, epochs = epochs, learn_rate = learn_rate, neurons = neurons)
#grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = 1, cv = 3)
#grid_result = grid.fit(X_train, Y_train)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))        #calculates the best parameters for training the data

#####################################################################

y = scaler_y.inverse_transform(f_model.predict(X_test))                #predicted the output for the test data input and rescaled the data to original values
print("Mean absolute error of Test data: %0.2f " % (mean_absolute_error(Y_test, y)))

s = np.sum((y -  Y_test.values.reshape(-1,1))**2)           
print(s)

plt.plot(y, 'g*', Y_test.values.reshape(-1,1), 'y*')           #plot for the original and predicted value of the test input
plt.title('Comparison between Predicted and Test Data')
plt.show()

pyplot.title('Loss / Mean Absolute Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


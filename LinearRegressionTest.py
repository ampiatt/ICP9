import pandas
from keras import metrics
from keras import regularizers
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pp

data = pandas.read_csv("kc_house_data.csv")
data['sale_yr'] = pandas.to_numeric(data.date.str.slice(0, 4))
data['sale_month'] = pandas.to_numeric(data.date.str.slice(4, 6))
data['sale_day'] = pandas.to_numeric(data.date.str.slice(6, 8))
house_data = pandas.DataFrame(data, columns=[ 'sale_yr','sale_month','sale_day',
        'bedrooms','bathrooms','sqft_living','sqft_lot','floors',
        'condition','grade','sqft_above','sqft_basement','yr_built',
        'zipcode','lat','long','sqft_living15','sqft_lot15','price'])
label_col = 'price'
print(house_data.describe())

x_train, x_test, y_train, y_test = train_test_split(house_data.iloc[:,0:18], house_data.iloc[:,18], test_size=0.25, random_state=87)


def create_model(x_size, y_size):
    seq_model = Sequential()
    seq_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    seq_model.add(Dense(50, activation="relu"))
    seq_model.add(Dense(y_size))
    print(seq_model.summary())
    seq_model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=[metrics.mae])
    return seq_model

used_model = create_model(x_train.shape[1], 1)
used_model.summary()

history_model = used_model.fit(x_train, y_train, batch_size=128, epochs=500, shuffle=True, verbose=0, validation_data=(x_test, y_test),)
trainscore = used_model.evaluate(x_train, y_train, verbose=0)
testscore = used_model.evaluate(x_test, y_test, verbose=0)
print('Training MAE: ', round(trainscore[1], 4), ', Train Loss:', round(trainscore[0], 4))
print('Testing MAE: ', round(testscore[1], 4), ', Test Loss: ', round(testscore[0], 4))

pp.plot(trainscore[1], testscore[1], 'bo')
pp.plot(trainscore[0], testscore[0], 'ro')
pp.show()

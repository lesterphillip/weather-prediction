#Import data pre-processing libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from numpy import array

# Import the KERAS Libraries to develop the model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

# Add seed to reproduce the same results every time
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

#Import plotting 
import matplotlib.pyplot as plt

# Plot designs
font = {'family' : 'Arial',
'weight' : 'normal',
'size' : 10}
plt.rc('font', **font)

# Importing the dataset
dataset = pd.read_csv('datasets/sample_radiation.csv')
dataset['0'] = pd.to_datetime(dataset['0'])
dataset = dataset.set_index('0')

# Split training set and test set
# Since this is a time series data, we can’t use the train_test_split as ordering is important
n_timestamp = 24 # number of hours to predict
train_days = 2196 # number of days to train from
testing_days = 480 # number of days to be predicted
n_epochs = 25
train_set = dataset[0:train_days].reset_index(drop=True)
test_set = dataset[train_days: train_days+testing_days].reset_index(drop=True)
training_set = train_set.iloc[:, 1:2].values
testing_set = test_set.iloc[:, 1:2].values
print("Choose LSTM model type")
print("1: Single cell")
print("2: Stacked")
print("3: Bidirectional")
model_type = input("Choose LSTM model type: ")
#model_type = 2

print("Apply a median filter and gaussian filter to the dataset? 1 = Yes, 0 = No")
#filter_on = 1 # apply a filter or not
filter_on = input("Apply a median filter and gaussian filter to the dataset?: ")
if filter_on == 1 or '1':
    dataset['Radiation'] = medfilt(dataset['Radiation'], 3)
    dataset['Radiation'] = gaussian_filter1d(dataset['Radiation'], 1.2)

# Apply scaler on the dataset for pre-processing
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(train_set)
testing_set_scaled = sc.fit_transform(test_set)

# Assigning time and radiation value into x and y
def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        if end_ix > len(sequence)-1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
X_train, y_train = data_split(training_set_scaled, n_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test, y_test = data_split(testing_set_scaled, n_timestamp)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Change model type value depending on model needs
if model_type == 1 or '1':
    # Single cell LSTM
    model = Sequential()
    model.add(LSTM(units = 50, activation='relu',input_shape = (X_train.shape[1], 1)))
    model.add(Dense(units = 1))
if model_type == 2 or '2':
    # Stacked LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
if model_type == 3 or '3':
    # Bidirectional LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))

# Train the model over the number of epochs
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = 32)
loss = history.history['loss']
epochs = range(len(loss))

# Predict next day based on model
y_predicted = model.predict(X_test)

# Remove pre-processing adjustments from scaler to get real values
y_predicted_descaled = sc.inverse_transform(y_predicted)
y_train_descaled = sc.inverse_transform(y_train)
y_test_descaled = sc.inverse_transform(y_test)
y_pred = y_predicted.ravel()
y_pred = [round(yx, 2) for yx in y_pred]
y_tested = y_test.ravel()

# Plot figures for results and discussion
plt.figure(figsize=(20,7))
plt.plot(dataset['Radiation'], color = 'black', linewidth=1, label = 'Truevalue')
plt.ylabel("Radiation")
plt.xlabel("Hour")
plt.title("All data")
plt.show()

plt.figure(figsize=(20,7))
plt.plot(y_test_descaled, color = 'black', linewidth=1, label = 'True value')
plt.plot(y_predicted_descaled, color = 'red', linewidth=1, label = 'Predicted')
plt.legend(frameon=False)
plt.ylabel("Radiation (W/m^2)")
plt.xlabel("Hour")
plt.title("Predicted data (20 days / 480 hours)")
plt.show()

plt.figure(figsize=(20,7))
plt.plot(y_test_descaled[-72:], color = 'black', linewidth=1, label = 'True value')
plt.plot(y_predicted_descaled[-72:], color = 'red', label = 'Predicted')
plt.legend(frameon=False)
plt.ylabel("Radiation (W/m^2)")
plt.xlabel("Hour")
plt.title("Predicted data (last 72 hours)")
plt.show()

plt.plot(epochs, loss, color='black')
plt.ylabel("Loss (MSE)")
plt.xlabel("Epoch")
plt.title("Training curve")
plt.show()

# Calculate errors
mse = mean_squared_error(y_test_descaled, y_predicted_descaled) 
r2 = r2_score(y_test_descaled, y_predicted_descaled) 
print("mse=" +str(round(mse,2))) 
print("r2=" + str(round(r2,2)))
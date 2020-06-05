# Imports
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Masking
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import SubANN, cm_pretty

# Config network
use_dropout = True
drop_rate = 0.2

# Get data
Inputs = np.loadtxt("Masked_Data/Inputs.csv", delimiter=",")
Outputs = np.loadtxt("Masked_Data/Outputs.csv", delimiter=",")
num_steps = Inputs.shape[1]

# Center and amplify each signal
#Inputs = SubANN.center_signals(Inputs)
#Inputs = SubANN.amplify_signals(Inputs, 10)

# Reshape inputs from 2D to 3D for the Masking and LSTM layers
Inputs = Inputs.reshape((Inputs.shape[0],Inputs.shape[1],1))

# Shuffle data
X_train, X_test, y_train, y_test = train_test_split(Inputs, Outputs, test_size=0.1, random_state=10)
print(SubANN.get_unique_counts_dict(y_train.argmax(axis=1)))

# Define the keras model
# Important info: The LSTM input layer must be 3D.
# The meaning of the 3 input dimensions are: samples, time steps, and features.
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
model = Sequential()
model.add(Masking(mask_value=-1000, input_shape=(num_steps,1)))
model.add(LSTM(500, input_shape=(num_steps,1), return_sequences=True))
model.add(LSTM(500, return_sequences=False))
if use_dropout:
    model.add(Dropout(drop_rate))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(model.summary())

# Fit the keras model on the dataset
history = model.fit(X_train, y_train, epochs=5, shuffle=True, validation_data=(X_test, y_test), batch_size=10, verbose=1)

# Evaluate the keras model
_, accuracy = model.evaluate(Inputs, Outputs)
print('Accuracy: %.2f' % (accuracy*100))

# Plots of the results
print(history.history.keys())  # list all data in history
SubANN.print_hist_parm(history, "categorical_accuracy")  # summarize history for accuracy
SubANN.print_hist_parm(history, "loss")  # summarize history for loss

# Plot confusion matrix
y_pred = model.predict(Inputs)
col = ["Entries", "Exits", "Lean outs"]
cm_pretty.plot_confusion_matrix_from_data(Outputs.argmax(axis=1),y_pred.argmax(axis=1),columns=col)
plt.show()
import numpy as np
import math 
import numpy
from csv import reader
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from sklearn import preprocessing as pre
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from sklearn.preprocessing import LabelEncoder

# Load data
data_train = pd.read_csv("features_csv_files/concatenated_file.csv", header=None) # Need to give the path to the file
label_train = pd.read_csv("label1.csv", header=None) # Need to give the path to the file

# Preprocess features
# data_train_scaled = (data_train - np.min(data_train)) / (np.max(data_train) - np.min(data_train))
# print (data_train_scaled)
print (data_train.shape)

X_train = np.array(data_train)
# X_train = X_train[:, :-1]  # Remove the last column if necessary
print (X_train)
print (X_train.shape)

# Preprocess labels
label_encoder = LabelEncoder()
label_train_encoded = label_encoder.fit_transform(label_train)
num_classes = len(label_encoder.classes_)

# Reshape features
num_videos = 958
frames_per_video = 45
num_features = 1027
Train_data = X_train.reshape(num_videos, frames_per_video, num_features)

# Split data into training and testing sets
trainX, testX, trainY, testY = train_test_split(Train_data, label_train_encoded, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True), input_shape=(frames_per_video, num_features)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=500, verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(testX, testY, verbose=1)
print("Accuracy:", accuracy)

# Make predictions
y_pred = model.predict(testX)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print evaluation metrics
print("Accuracy:", accuracy_score(testY, y_pred_classes))
print("Confusion Matrix:\n", confusion_matrix(testY, y_pred_classes))
print("Classification Report:\n", classification_report(testY, y_pred_classes))
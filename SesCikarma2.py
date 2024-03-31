import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau

# Directory where the audio files are located
data_dir = "Hasan"

# Array to store feature vectors and labels
features = []
labels = []

# Process all audio files
for filename in os.listdir(data_dir):
    if filename.endswith('.wav'):
        # Load audio file
        filepath = os.path.join(data_dir, filename)
        audio_data, sr = librosa.load(filepath)
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)  
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        features.append(mfccs_processed)
        labels.append(filename.split('-')[0])  

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Reshape the dataset (to fit the input shape of CNN)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convert labels to categorical
num_classes = len(np.unique(labels))
y_train_categorical = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)

# CNN model
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(256, 3, activation='relu')) 
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Adding a callback for reducing learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Training the model
model.fit(X_train, y_train_categorical, epochs=150, batch_size=64, validation_split=0.2, callbacks=[reduce_lr])

# Evaluating the model performance
score = model.evaluate(X_test, y_test_categorical, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model
model.save('speech_recognition_cnn_model_complex4.h5')

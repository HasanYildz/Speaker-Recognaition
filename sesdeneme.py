import librosa
import numpy as np
from keras.models import load_model

model = load_model('speech_recognition_cnn_model_complex4.h5')

def predict_hasan(audio_file):
    audio_data, sr = librosa.load(audio_file)
    
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40) 
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    audio_input = mfccs_processed.reshape(1, mfccs_processed.shape[0], 1)
    
    prediction = model.predict(audio_input)
    predicted_label = np.argmax(prediction)
    
    return predicted_label

# Test
audio_file_path = "Recording (19).wav"  
predicted_class = predict_hasan(audio_file_path)

if predicted_class == 0:
    print("The voice belongs to Hasan.")
else:
    print("The voice does not belong to Hasan.")


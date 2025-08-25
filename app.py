import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import librosa
import librosa.display
from imblearn.over_sampling import RandomOverSampler
import os
features_30=pd.read_csv('C:/Users/DELL/Desktop/ML_learning_internship/Music_Genre_Classification_Task_4/Dataset/features_3_sec.csv')
genre_original=('C:/Users/DELL/Desktop/ML_learning_internship/Music_Genre_Classification_Task_4/Dataset/genres_original')

print("Feauters 30 seconds Dataframe: \n")
print(features_30.head())
print("Dataframe Shape: ",features_30.shape)
print("Dataframe Information: ",features_30.info())
print("Dataframe statistics: ",features_30.describe())
print("number of null rows", features_30.isnull().sum())
print("number of duplicate rows", features_30.duplicated().sum())

dataframe_copy=features_30.copy()
dataframe_copy=dataframe_copy.drop(['filename','length','label'],axis=1)
X=dataframe_copy
Y=features_30['label']

data=[]
spectrograms = []
labels = []

for folder in os.listdir(genre_original):
    genre_classification = os.path.join(genre_original, folder)
    count = 0

    for audio in os.listdir(genre_classification):
        if count >=1600:
            break

        audio_file = os.path.join(genre_classification, audio)
        try:
            y, sr = librosa.load(audio_file, sr=22050,duration=30)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)  # Pitch shift
            y_stretch = librosa.effects.time_stretch(y, rate=0.8)  # Time stretch
            noise = np.random.normal(0, 0.005, y.shape)  # Add noise
            y_noise = y + noise
            log_S = librosa.power_to_db(S, ref=np.max)
            log_S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128))
            log_S = librosa.util.fix_length(log_S, size=1292, axis=1)
            spectrograms.append(log_S)
            labels.append(folder)

            # Extract features
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
            mfccs_orig = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_pitch = librosa.feature.mfcc(y=y_pitch, sr=sr, n_mfcc=40)
            mfccs_stretch = librosa.feature.mfcc(y=y_stretch, sr=sr, n_mfcc=40)
            mfccs_noise = librosa.feature.mfcc(y=y_noise, sr=sr, n_mfcc=40)
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
            flatness = np.mean(librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=512))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            loudness = np.mean(librosa.feature.rms(y=y))
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y=y))
            rhythm_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]


            row = {
                "centroid": centroid,
                "bandwidth": bandwidth,
                "rolloff": rolloff,
                "zcr": zero_crossing,
                "flatness": flatness,
                "spectral_contrast": spectral_contrast,
                "chroma": chroma,
                "loudness": loudness,
                "rhythm_strength": rhythm_strength,
                "tempo": tempo
            }

            for i, mfcc in enumerate(mfccs):
                row[f"mfcc{i + 1}"] = mfcc

            row["genre"] = folder
            data.append(row)
            count += 1

        except Exception as e:
            print(f"Skipping {audio_file}: {e}")
print(len(spectrograms), len(labels))

music_dataframe=pd.DataFrame(data)
print(music_dataframe.head())
music_dataframe.to_csv("music_dataframe.csv")

print("model depends on labrosa")
x=music_dataframe.drop("genre",axis=1)
y=music_dataframe["genre"]
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    x, y, test_size=0.2, random_state=42,stratify=y
)

scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_test1 = scaler.transform(X_test1)

# Random Forest model
model1=RandomForestClassifier(n_estimators=300, max_depth=50, random_state=42)
model1.fit(X_train1,y_train1)

y_pred1 = model1.predict(X_test1)
print("Accuracy:", accuracy_score(y_test1, y_pred1))
print(classification_report(y_test1, y_pred1,zero_division=0))
print(confusion_matrix(y_test1, y_pred1))



print("random forest model")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)





model=RandomForestClassifier(n_estimators=300, max_depth=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
num_features = X_train1.shape[1]
X_train_cnn = X_train1.reshape(X_train1.shape[0], X_train1.shape[1], 1)
X_test_cnn  = X_test1.reshape(X_test1.shape[0], X_test1.shape[1], 1)



#cnn model
fixed_length = 1292 # number of time frames for all spectrograms
X_cnn = []

for spec in spectrograms:
    if spec.shape[1] < fixed_length:

        pad_width = fixed_length - spec.shape[1]
        spec_padded = np.pad(spec, ((0,0),(0,pad_width)), mode='constant')
    else:

        spec_padded = spec[:, :fixed_length]
    X_cnn.append(spec_padded)

X_cnn = np.array(X_cnn, dtype=np.float32)
assert len(X_cnn) == len(labels), "Number of spectrograms and labels must match!"
y_cnn = np.array(labels)

X_cnn = X_cnn / np.max(X_cnn)
X_cnn = X_cnn[..., np.newaxis]
print("x ccn lenght and y ccn length")
print(len(X_cnn), len(y_cnn))

encoder = LabelEncoder()
y_cnn_enc = encoder.fit_transform(y_cnn)

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y_cnn_enc, test_size=0.2, random_state=42, stratify=y_cnn_enc
)

X_train_cnn_flat = X_train_cnn.reshape(len(X_train_cnn), -1)
ros = RandomOverSampler(random_state=42)
X_train_cnn, y_train_cnn = ros.fit_resample(X_train_cnn_flat, y_train_cnn)
X_train_cnn = X_train_cnn.reshape(-1, 128, 1292, 1)

print("CNN model")
print("CNN model input shape:", X_train_cnn.shape)
unique_genre=music_dataframe["genre"].unique()
print("number of unique genres: ", unique_genre)


num_classes = len(encoder.classes_)
model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, fixed_length,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model2.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)
history = model2.fit(X_train_cnn, y_train_cnn,
                     validation_split=0.2,
                     epochs=100,
                     batch_size=32,
                     callbacks=[callback,reduce_lr])

# Evaluate CNN
test_loss, test_acc = model2.evaluate(X_test_cnn,y_test_cnn, verbose=2)
print('\nTest accuracy:', test_acc)

y_pred_probs = model2.predict(X_test_cnn)
y_pred1 = np.argmax(y_pred_probs, axis=1)

#Evaluation

print("CNN Accuracy:", accuracy_score(y_test_cnn,y_pred1))
print(classification_report(y_test_cnn,y_pred1))
print(confusion_matrix(y_test_cnn,y_pred1))


cm = confusion_matrix(y_test_cnn, y_pred1,labels=np.arange(len(encoder.classes_)))
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - CNN")
plt.show()



MUSIC GENRE CLASSIFICATION

Project Description:
This project classifies music tracks into their respective genres using audio feature extraction and machine learning models. 
Both Random Forest and Convolutional Neural Network (CNN) models are implemented for comparison.

Dataset:
- Original dataset directory: 'genres_original/'
- Each folder corresponds to a music genre (e.g., blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock).
- Maximum 1600 audio files per genre are processed.
- Audio files are 30 seconds long, sampled at 22,050 Hz.
- Total number of samples used for training/testing depends on the number of files per genre.

Feature Extraction:
- CSV-based features (features_3_sec.csv) include MFCCs, spectral centroid, bandwidth, rolloff, flatness, spectral contrast, chroma, RMS, zero crossing rate, onset strength, and tempo.
- Spectrogram-based features extracted using Librosa for CNN.
- Data augmentation for CNN: pitch shift, time stretch, added noise.
- Spectrograms are computed using melspectrogram and converted to log scale.

Data Preprocessing:
- Dropped irrelevant columns: 'filename', 'length', 'label' for Random Forest
- Standardization using StandardScaler
- Label encoding for genre labels
- Train-test split: 80% training, 20% testing
- For CNN:
    - Fixed spectrogram length: 1292 frames
    - Normalized and reshaped to (128, 1292, 1)
    - Oversampling using RandomOverSampler to balance classes

Random Forest Models:

1. **CSV-based Random Forest (features from features_3_sec.csv)**
   - Input: Precomputed CSV features
   - Number of samples: ~[number of rows in features_3_sec.csv]
   - Parameters: n_estimators=300, max_depth=50
   - Trained on 80% of the dataset
   - Accuracy: 88%
   - Outputs: Accuracy, Classification Report, Confusion Matrix
   - **Best performing model in this project.**

2. **Librosa-based Random Forest (features extracted from audio)**
   - Input: Librosa-extracted features (MFCCs, spectral, temporal)
   - Number of samples: matches number of processed audio files (~1600 per genre max)
   - Parameters: n_estimators=300, max_depth=50
   - Accuracy: 66.5%
   - Outputs: Accuracy, Classification Report, Confusion Matrix

CNN Model:
- Input: Spectrogram images (128, 1292, 1)
- Architecture:
    1. Conv2D (16 filters, 3x3, ReLU)
    2. MaxPooling2D (2x2)
    3. Conv2D (32 filters, 3x3, ReLU)
    4. MaxPooling2D (2x2)
    5. Conv2D (64 filters, 3x3, ReLU)
    6. MaxPooling2D (2x2)
    7. Flatten
    8. Dense 128 (ReLU)
    9. Dropout 0.4
    10. Dense num_classes (Softmax)
- Optimizer: Adam, learning rate 1e-4
- Loss: Sparse Categorical Crossentropy
- Callbacks:
    - EarlyStopping (monitor val_loss, patience 5)
    - ReduceLROnPlateau (factor 0.3, patience 3)
- Epochs: 100, Batch size: 32
- Accuracy: 63%

Evaluation:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- CNN confusion matrix visualized using Seaborn heatmap

Usage Instructions:
1. Place dataset folders under 'genres_original/'.
2. Place 'features_3_sec.csv' in the project directory.
3. Run the Python script to extract features, train models, and evaluate.
4. Libraries required:
   - numpy, pandas, matplotlib, seaborn
   - scikit-learn
   - imbalanced-learn
   - tensorflow, keras
   - librosa

Outputs:
- music_dataframe.csv (Librosa features)
- Random Forest model evaluation metrics (CSV-based and Librosa-based)
- CNN model evaluation metrics and confusion matrix

Acknowledgments:
- Librosa library for audio processing and feature extraction
- Scikit-learn for Random Forest implementation
- TensorFlow/Keras for CNN implementation





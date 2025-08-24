This project focuses on classifying music genres using two different machine learning approaches:

Random Forest Classifier (with Librosa Features)

We extracted audio features from the GTZAN dataset using the Librosa library.

Features include: chroma, spectral centroid, spectral bandwidth, spectral roll-off, zero-crossing rate, MFCCs, etc.

A Random Forest classifier was trained on these extracted features.

First experiment: with a small number of samples (around 40), accuracy reached ~81%.

When increasing the dataset size to more samples (e.g., 200), accuracy dropped to around 67% because the model faced more variety and complexity.

Convolutional Neural Network (CNN)

Instead of only extracting handcrafted features, we also built a deep learning model (CNN) to learn features directly from spectrogram-like representations.

We tried different architectures. The initial CNN was too heavy and caused very long training times.

To improve this:

Reduced the number of filters (16, 32, 64 instead of very large ones).

Added Dropout layers to reduce overfitting.

Used GlobalAveragePooling2D instead of Flatten to reduce parameters.

Added Early Stopping to prevent wasting time when validation accuracy stops improving.

With these optimizations, training is faster, more stable, and avoids heavy overfitting.

Dataset

Based on GTZAN Music Genre Dataset.

We experimented with different numbers of samples:

40 samples: quick experiment, higher accuracy (but less generalization).

200 samples: more realistic, slower training, accuracy lower due to higher challenge.

Plan to scale further (e.g., 300 samples) while applying regularization to balance performance and overfitting.

Current Status

Random Forest + Librosa features gave a strong baseline.

CNN approach is still training with optimizations (reduced complexity, dropout, early stopping).

Next step: compare results and decide whether feature-based ML (Random Forest) or feature-learning DL (CNN) generalizes better.

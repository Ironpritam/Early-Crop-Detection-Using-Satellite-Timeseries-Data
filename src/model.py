import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from .dataset_preprocessing import calculate_ndvi


class Inception(tf.keras.layers.Layer):
    """
    1D Inception Module for multi-scale temporal-spectral feature extraction.
    Applies parallel 1x1, 3x3, 5x5 Conv1D filters and MaxPool1D operations.
    """
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

        self.b1_1 = Conv1D(c1, 1, activation='relu', name='b1_conv1x1')

        self.b2_1 = Conv1D(c2[0], 1, activation='relu', name='b2_conv1x1')
        self.b2_2 = Conv1D(c2[1], 3, padding='same', activation='relu', name='b2_conv3x3')

        self.b3_1 = Conv1D(c3[0], 1, activation='relu', name='b3_conv1x1')
        self.b3_2 = Conv1D(c3[1], 5, padding='same', activation='relu', name='b3_conv5x5')

        self.b4_1 = MaxPool1D(3, 1, padding='same', name='b4_maxpool')
        self.b4_2 = Conv1D(c4, 1, activation='relu', name='b4_conv1x1')

    def call(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return Concatenate()([b1, b2, b3, b4])

    def get_config(self):
        config = super().get_config()
        config.update({
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
            'c4': self.c4,
        })
        return config


class Crop_BB:
    """
    Crop Backbone neural network generator based on stacked 1D Inception blocks.
    """
    def b1(self):
        return Sequential([
            Conv1D(32, 1, activation='relu', name='b1_conv1x1'),
            Conv1D(64, 3, padding='same', activation='relu', name='b1_conv3x3'),
            MaxPool1D(pool_size=3, strides=2, padding='same', name='b1_maxpool')
        ])

    def b2(self):
        return Sequential([
            Inception(64, (34, 64), (8, 16), 16),
            MaxPool1D(pool_size=3, strides=2, padding='same', name='b2_maxpool')
        ])

    def b3(self, num_classes=3):
        return Sequential([
            Inception(64, (40, 80), (16, 32), 32),
            Inception(96, (96, 192), (24, 64), 64),
            Conv1D(64, 1, activation='relu', name='b3_conv1x1'),
            GlobalAveragePooling1D(name='global_avg_pool'),
            Dense(27, activation='relu', name='dense_layer'),
            Dense(num_classes, activation='softmax', name='dense_output')
        ])


def build_model(input_shape=None, num_classes=3):
    """
    Constructs and compiles the 1D Inception model for crop detection.
    """
    crops_bb = Crop_BB()
    model = Sequential([crops_bb.b1(), crops_bb.b2(), crops_bb.b3(num_classes)])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_prebuilt_model(model_path: str):
    """
    Loads a saved H5/Keras model checkpoint with custom layer handling.
    """
    custom_objects = {'Inception': Inception}
    return load_model(model_path, custom_objects=custom_objects)


def load_data(csv_folder: str):
    """
    Loads pixel time-series dataset from nested class folders containing CSV files.
    Returns:
        X: Numpy array of NDVI time-series features.
        y: Numpy array of class indices.
    """
    X = []
    y = []
    class_labels = {}
    next_class_idx = 0

    if not os.path.exists(csv_folder):
        raise FileNotFoundError(f"Dataset folder '{csv_folder}' does not exist.")

    folder_names = sorted([d for d in os.listdir(csv_folder) if os.path.isdir(os.path.join(csv_folder, d))])
    for folder_name in folder_names:
        class_labels[folder_name] = next_class_idx
        next_class_idx += 1
        subfolder_path = os.path.join(csv_folder, folder_name)
        
        for csv_file in os.listdir(subfolder_path):
            if csv_file.endswith('.csv'):
                csv_path = os.path.join(subfolder_path, csv_file)
                df = pd.read_csv(csv_path)
                
                if 'NDVI' in df.columns:
                    ndvi = df[['NDVI']].values
                elif 'Band_6' in df.columns and 'Band_8' in df.columns:
                    red = df[['Band_6']].values
                    nir = df[['Band_8']].values
                    ndvi = calculate_ndvi(red, nir)
                else:
                    # Fallback to available numerical features
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    ndvi = df[numeric_cols].values
                
                X.append(ndvi)
                y.append(class_labels[folder_name])

    return np.array(X), np.array(y)


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Normalizes feature values using MinMaxScaler and one-hot encodes labels.
    """
    X_flat = X.reshape(X.shape[0], -1)
    scaler = MinMaxScaler()
    X_normalized_flat = scaler.fit_transform(X_flat)
    X_normalized = X_normalized_flat.reshape(X.shape)
    y_cat = to_categorical(y)

    return train_test_split(X_normalized, y_cat, test_size=test_size, random_state=random_state)


def test_model(model, X_test, y_test):
    """
    Evaluates the model on test data and calculates accuracy, precision, recall, and F1 metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    if y_test.ndim > 1:
        y_true_classes = np.argmax(y_test, axis=1)
    else:
        y_true_classes = y_test

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

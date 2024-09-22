import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc



class Inception(tf.keras.Model):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_1 = tf.keras.layers.Conv1D(c1, 1, activation='relu', name='b1_conv1x1')

        self.b2_1 = tf.keras.layers.Conv1D(c2[0], 1, activation='relu', name='b2_conv1x1')
        self.b2_2 = tf.keras.layers.Conv1D(c2[1], 3, padding='same', activation='relu', name='b2_conv3x3')

        self.b3_1 = tf.keras.layers.Conv1D(c3[0], 1, activation='relu', name='b3_conv1x1')
        self.b3_2 = tf.keras.layers.Conv1D(c3[1], 5, padding='same', activation='relu', name='b3_conv5x5')

        self.b4_1 = tf.keras.layers.MaxPool1D(3, 1, padding='same', name='b4_maxpool')
        self.b4_2 = tf.keras.layers.Conv1D(c4, 1, activation='relu', name='b4_conv1x1')

    def call(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return tf.keras.layers.Concatenate()([b1, b2, b3, b4])

# Building classifier model using convolution layers.
class Crop_BB():
    def b1(self):
        return tf.keras.Sequential([
              tf.keras.layers.Conv1D(32, 1, activation='relu', name='b1_conv1x1'),
              tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu', name='b1_conv3x3'),
              tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same', name='b1_maxpool')])
    def b2(self):
        return tf.keras.Sequential([
            Inception(64, (34, 64), (8, 16), 16),
            tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same', name='b2_maxpool')])
    def b3(self):
        return tf.keras.Sequential([
            Inception(64, (40, 80), (16, 32), 32),
            Inception(96, (96, 192), (24, 64), 64),
            tf.keras.layers.Conv1D(64, 1, activation='relu', name='b3_conv1x1'),
            tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool'),
            tf.keras.layers.Dense(27, activation='relu', name='dense_layer'),
            tf.keras.layers.Dense(3, activation='softmax', name='dense_output')])


def test_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = (np.argmax(y_pred, axis=1)+2) % 3

    # Convert one-hot encoded labels to single integer labels
    y_true_classes = np.argmax(y_test, axis=1)
    print(f"Y_True:{y_true_classes}\n")
    print(f"Y_Pred:{y_pred_classes}")
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

    return accuracy, precision, recall, f1


def load_prebuilt_model(model_path):
    # Define custom objects dictionary to include the custom layers/classes
    custom_objects = {'Inception': Inception}

    # Load the model using the custom_objects parameter
    return load_model(model_path, custom_objects=custom_objects)

def calculate_ndvi(red, nir):
    fillZeroInRed = np.mean(red)
    red[red == 0] = 1

    fillZeroInNir = np.mean(nir)
    nir[nir == 0] = 1

    ndvi = (nir.astype(float) - red.astype(float)) / (nir.astype(float) + red.astype(float))

    # Replace negative NDVI values with 0
    #ndvi = np.where(ndvi < 0, 0, ndvi)

    return ndvi

# Step 1: Data Loading
def load_data(csv_folder):
    X = []
    y = []
    class_labels = {}  # Dictionary to map class names to numerical indices
    next_class_idx = 0

    for folder_name in os.listdir(csv_folder):
        if os.path.isdir(os.path.join(csv_folder, folder_name)):
            class_labels[folder_name] = next_class_idx
            next_class_idx += 1

            for csv_file in os.listdir(os.path.join(csv_folder, folder_name)):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(csv_folder, folder_name, csv_file)
                    df = pd.read_csv(csv_path)
                    # Assuming 'NDVI' column contains the NDVI index data
                    red = df[['Band_6']].values
                    nir = df[['Band_8']].values
                    ndvi = calculate_ndvi(red, nir)
                    # ndvi_data = df[['NDVI']].values.reshape(-1, 8, 8, 1)  # Assuming 8x8 pixels
                    X.append(ndvi)
                    y.append(class_labels[folder_name])  # Use numerical class index
    return np.array(X), np.array(y)

# Step 2: Data Preprocessing
def preprocess_data(X, y):
    # Reshape X to 2D array for normalization
    X_flat = X.reshape(X.shape[0], -1)

    # Min-max normalization
    scaler = MinMaxScaler()
    X_normalized_flat = scaler.fit_transform(X_flat)

    # Reshape X back to its original shape
    X_normalized = X_normalized_flat.reshape(X.shape)

    # Convert labels to categorical
    y = to_categorical(y)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

# Step 3: Model Building
def build_model(input_shape, num_classes):
    crops_bb = Crop_BB()
    model = tf.keras.Sequential([crops_bb.b1(), crops_bb.b2(), crops_bb.b3()])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Model Training
def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    checkpoint_path = "/saved_models/best_model_epoch_{epoch:02d}.h5"

    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val), epochs=epochs, callbacks=[checkpoint_callback, early_stopping_callback])

# Step 5: Model Evaluation
def evaluate_model(model, X_val, y_val):
    loss, accuracy = model.evaluate(X_val, y_val)
    print("Validation Loss:", loss)
    print("Validation Accuracy:", accuracy)

# Step 6: Prediction
def make_predictions(model, X):
    predictions = model.predict(X)
    # Convert predictions to labels
    labels = np.argmax(predictions, axis=1)
    return labels

# Main function


def main():
    csv_folder = '/content/drive/My Drive/ACPS/Dataset_csv'
    X, y = load_data(csv_folder)
    X_train, X_val, y_train, y_val = preprocess_data(X, y)

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y))
    model = build_model(input_shape, num_classes)

    checkpoint_path = "/saved_models/best_model_epoch_{epoch:02d}.h5"
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val), epochs=100,
              callbacks=[checkpoint_callback, early_stopping_callback])

    print("__" * 50)
    print("\n \nEvaluating model ... \n\n")
    print("__" * 50)
    evaluate_model(model, X_val, y_val)

    return model

if __name__ == "__main__":
    model = main()
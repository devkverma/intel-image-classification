
# Landscape Classification with TensorFlow

This project implements a Convolutional Neural Network (CNN) to classify landscapes using images. The model is trained on a dataset of landscape images and can predict the category of a given image.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting the Dataset](#getting-the-dataset)
- [Model Training](#model-training)
- [Usage](#usage)
- [Additional Information](#additional-information)

## Overview
The project utilizes TensorFlow and Keras to create a CNN model that processes images and learns to classify them into different categories. The model is capable of recognizing various landscape types based on features extracted from the images.

## Prerequisites
Make sure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- scikit-learn
- Matplotlib
- NumPy
- Kaggle

You can install the required packages using pip:
```bash
pip install tensorflow keras opencv-python scikit-learn matplotlib numpy kaggle
```

## Getting the Dataset

To download the Intel Image Classification dataset using the Kaggle API, follow these steps:

### Step 1: Set Up Kaggle API

1. **Create a Kaggle Account**:
   - If you haven't already, go to [Kaggle's website](https://www.kaggle.com/) and create an account.

2. **Get Your API Token**:
   - Log in to your Kaggle account.
   - Click on your profile picture in the top right corner and select **Account**.
   - Scroll down to the **API** section.
   - Click on **Create New API Token**. This action will download a file named `kaggle.json` to your computer.

### Step 2: Configure the Kaggle API

3. **Install the Kaggle Library**:
   Open your terminal or command prompt and install the Kaggle library if you haven't already:
   ```bash
   pip install kaggle
   ```

4. **Place the `kaggle.json` File**:
   - **For Windows**: Move the `kaggle.json` file to `C:\Users\<YourUsername>\.kaggle\`.
   - **For macOS/Linux**: Move it to `~/.kaggle/`.

   You can use the following commands to create the directory and move the file (assuming you're on macOS/Linux):
   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/your/downloaded/kaggle.json ~/.kaggle/
   ```

5. **Set Permissions (macOS/Linux)**:
   Ensure that the `kaggle.json` file has the correct permissions so that it's secure:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 3: Download the Dataset

6. **Run the Download Command**:
   Use the following command in your terminal to download the Intel Image Classification dataset:
   ```bash
   kaggle datasets download -d puneet6060/intel-image-classification
   ```

7. **Unzip the Dataset**:
   The dataset will be downloaded as a zip file named `intel-image-classification.zip`. You need to unzip it. You can do this using the command line:

   For Linux or macOS:
   ```bash
   unzip intel-image-classification.zip
   ```

   For Windows, you can use the built-in file explorer to right-click on the zip file and select **Extract All**.

## Model Training

Once you have the dataset, you can train the model using the provided script. The script does the following:

1. **Load and Preprocess the Images**:
   - Load images from the training and testing directories.
   - Resize images to 100x100 pixels.
   - Normalize pixel values to the range [0, 1].

2. **Create and Compile the Model**:
   - Define a CNN architecture with convolutional, pooling, and fully connected layers.
   - Compile the model using Adam optimizer and sparse categorical crossentropy loss.

3. **Train the Model**:
   - Fit the model on the training data with early stopping and learning rate reduction callbacks.

4. **Save the Model**:
   - Save the trained model to a file named `model.pkl`.

### Example of Model Training
Here's a snippet of the code used to train the model:
```python
# Model definition
model = Sequential()
model.add(Input((100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(6, activation='softmax'))

# Compile and fit the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val), epochs=20)
```

## Usage
To use the trained model for predictions, load the `model.pkl` file and pass images through the model.

### Example of Loading and Using the Model
```python
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Use the model to make predictions
predictions = model.predict(new_images)
```

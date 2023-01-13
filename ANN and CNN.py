import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# Appending the images of different folders into one array
images = []
labels = []


def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            labels.append(folder[-1])
    return images, labels


def resize(height, width, images):
    dim = (width, height)
    new = []
    for img in images:
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        new.append(resized)
    return new


def scale(flag, resized_images):
    # 1 means grayscale, 0 means RGB
    if flag:
        grayscale = []
        for img in resized_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayscale.append(gray)
        grayscale = np.asarray(grayscale, dtype=object)
        return grayscale
    else:
        rgbscale = []
        for img in resized_images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgbscale.append(rgb)

        rgbscale = np.array(rgbscale, dtype=object)
        return rgbscale


def ANN_model(X_train, X_test, Y_train, Y_test):
    # Define the model
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model1.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
                   metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                            tf.keras.metrics.AUC()])

    # Create a KFold object with 5 folds
    kfold = KFold(n_splits=5, shuffle=False, random_state=None)

    # Initialize lists to store the results for each fold
    train_acc = []
    val_acc = []
    test_acc = []

    train_loss = []
    val_loss = []
    test_loss = []

    precision = []

    recall = []

    auc = []

    # Loop through the folds
    for train_index, test_index in kfold.split(X_train):
        # Split the data into training and validation sets
        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
        Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[test_index]

        # Fit the model on the training data
        model1.fit(X_train_fold, Y_train_fold, epochs=50, batch_size=32, verbose=0)

        # Extract the training, validation and testing accuracy from the evaluate method
        fold_val_loss, fold_val_acc, _, _, _ = model1.evaluate(X_val_fold, Y_val_fold, verbose=0)
        fold_train_loss, fold_train_acc, _, _, _ = model1.evaluate(X_train_fold, Y_train_fold, verbose=0)
        fold_test_loss, fold_test_acc, fold_test_pre, fold_test_rec, fold_test_auc = model1.evaluate(X_test, Y_test,
                                                                                                     verbose=0)

        # Append the results to the lists
        train_acc.append(fold_train_acc)
        val_acc.append(fold_val_acc)
        test_acc.append(fold_test_acc)

        train_loss.append(fold_train_loss)
        val_loss.append(fold_val_loss)
        test_loss.append(fold_test_loss)

        precision.append(fold_test_pre)

        recall.append(fold_test_rec)

        auc.append(fold_test_auc)

    # Calculate the mean of the cross-validation scores
    print("Mean Loss and Accuracy for training: ", [np.mean(train_loss), np.mean(train_acc)])
    print("Mean Loss and Accuracy for validation: ", [np.mean(val_loss), np.mean(val_acc)])
    print("Mean Loss and Accuracy for testing: ", [np.mean(test_loss), np.mean(test_acc)])
    print("Mean Precision score: ", np.mean(precision))
    print("Mean Recall score: ", np.mean(recall))
    print("Mean AUC score: ", np.mean(auc))


def CNN_model(X_train, X_test, Y_train, Y_test):
    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')

    ])
    # Compile the model
    model2.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
                   metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                            tf.keras.metrics.AUC()])

    # Create a KFold object with 5 folds
    kfold = KFold(n_splits=5)

    # Initialize lists to store the results for each fold
    train_acc = []
    val_acc = []
    test_acc = []

    train_loss = []
    val_loss = []
    test_loss = []

    precision = []

    recall = []

    auc = []

    # Loop through the folds
    for train_index, test_index in kfold.split(X_train):
        # Split the data into training and validation sets
        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
        Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[test_index]

        # Fit the model on the training data
        model2.fit(X_train_fold, Y_train_fold, epochs=2, batch_size=32, verbose=0)

        # Extract the training, validation and testing accuracy from the evaluate method
        fold_val_loss, fold_val_acc, _, _, _ = model2.evaluate(X_val_fold, Y_val_fold, verbose=0)
        fold_train_loss, fold_train_acc, _, _, _ = model2.evaluate(X_train_fold, Y_train_fold, verbose=0)
        fold_test_loss, fold_test_acc, fold_test_pre, fold_test_rec, fold_test_auc = model2.evaluate(X_test, Y_test,
                                                                                                     verbose=0)

        # Append the results to the lists
        train_acc.append(fold_train_acc)
        val_acc.append(fold_val_acc)
        test_acc.append(fold_test_acc)

        train_loss.append(fold_train_loss)
        val_loss.append(fold_val_loss)
        test_loss.append(fold_test_loss)

        precision.append(fold_test_pre)

        recall.append(fold_test_rec)

        auc.append(fold_test_auc)
    # Calculate the mean of the cross-validation scores

    print("Mean Loss and Accuracy for training: ", [np.mean(train_loss), np.mean(train_acc)])
    print("Mean Loss and Accuracy for validation: ", [np.mean(val_loss), np.mean(val_acc)])
    print("Mean Loss and Accuracy for testing: ", [np.mean(test_loss), np.mean(test_acc)])
    print("Mean Precision score: ", np.mean(precision))
    print("Mean Recall score: ", np.mean(recall))
    print("Mean AUC score: ", np.mean(auc))


dir = '/Dataset/'

zero = dir + "/0"
one = dir + "/1"
two = dir + "/2"
three = dir + "/3"
four = dir + "/4"
five = dir + "/5"
six = dir + "/6"
seven = dir + "/7"
eight = dir + "/8"
nine = dir + "/9"

arr = [zero, one, two, three, four, five, six, seven, eight, nine]

plt.imshow(mpimg.imread('/Users/youssefsoultan/Desktop/Images/Dataset/1/IMG_1119.JPG'))

for folder in arr:
    images, label = load_images_from_folder(folder)

new_size = resize(64, 64, images)
gray_scale = scale(1, new_size)
rgb_scale = scale(0, new_size)

X_gray = np.divide(gray_scale, 255)
X_gray = np.array(X_gray).astype('float32')

Y_gray = np.array(label).astype('float32')
Y_gray = tf.keras.utils.to_categorical(label, 10)

X_rgb = np.divide(rgb_scale, 255)
X_rgb = np.array(X_rgb).astype('float32')

Y_rgb = np.array(label).astype('float32')
Y_rgb = tf.keras.utils.to_categorical(label, 10)

# Split the data into training and testing sets
X_train_gray, X_test_gray, Y_train_gray, Y_test_gray = train_test_split(X_gray, Y_gray, test_size=0.2)
X_train_rgb, X_test_rgb, Y_train_rgb, Y_test_rgb = train_test_split(X_rgb, Y_rgb, test_size=0.2)

print("Total number of gray images = " , X_gray.shape[0])
print("number of training examples for gray images = ", X_train_gray.shape[0])
print("number of test examples for gray images = ", X_test_gray.shape[0])
print("X_train shape: ", X_train_gray.shape)
print("Y_train shape: ", Y_train_gray.shape)
print("X_test shape: ", X_test_gray.shape)
print("Y_test shape: ", Y_test_gray.shape)

print('<==================================================================>')
print("Total number of RGB images = ", X_rgb.shape[0])
print("number of training examples for RGB images = ", X_train_rgb.shape[0])
print("number of test examples for RGB images = ", X_test_rgb.shape[0])
print("X_train shape: ", X_train_rgb.shape)
print("Y_train shape: ", Y_train_rgb.shape)
print("X_test shape: ", X_test_rgb.shape)
print("Y_test shape: ", Y_test_rgb.shape)

print("ANN Results: ")
ANN_model(X_train_gray, X_test_gray, Y_train_gray, Y_test_gray)
print('<==================================================================>')
print("CNN Results: ")
CNN_model(X_train_rgb, X_test_rgb, Y_train_rgb, Y_test_rgb)

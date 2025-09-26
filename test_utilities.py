import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def test_plot_images():
    # Test the plot_images utility with dummy data
    images = np.random.rand(4, 256, 256, 3)
    labels = [0, 1, 0, 1]
    def plot_images(images, labels=None, ncols=4):
        fig, ax = plt.subplots(1, ncols, figsize=(15, 5))
        for idx, img in enumerate(images[:ncols]):
            ax[idx].imshow(img.astype(np.float32))
            if labels is not None:
                ax[idx].set_title(str(labels[idx]))
            ax[idx].axis('off')
        plt.show()
    plot_images(images, labels)

def test_build_model():
    # Test the model building function
    def build_model(input_shape=(256, 256, 3)):
        model = Sequential()
        model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(16, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model
    model = build_model()
    assert isinstance(model, Sequential)
    assert model.input_shape[1:] == (256, 256, 3)
    print("Model built successfully.")

if __name__ == "__main__":
    test_plot_images()
    test_build_model()
    print("All utility tests passed.")

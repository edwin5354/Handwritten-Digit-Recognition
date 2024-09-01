# Import relevant libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Variables
num_class = 10
batch_size = 32
epochs = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# Load the data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Reshape the training and testing data for CNN model
train_X = train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)
test_X = test_X.reshape(test_X.shape[0], img_rows, img_cols, 1)

# Feature scaling (divide by 255)
test_X = test_X.astype('float32')
train_X = train_X.astype('float32')

test_X /= 255.
train_X /= 255.

# One-Hot-Encoding on train_y, test_y data
train_y = tf.keras.utils.to_categorical(train_y, num_class)
test_y = tf.keras.utils.to_categorical(test_y, num_class)

# Create a CNN model for training the data
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape= input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))

    return model

CNN_model = create_model()
CNN_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

CNN_history = CNN_model.fit(train_X, train_y, batch_size= batch_size, epochs= epochs, validation_data=(test_X, test_y))

# plot model accuracy and loss
def accuracy():    
    epochs_len = range(1, epochs + 1)
    plt.plot(epochs_len, CNN_history.history['accuracy'], 'bx-', label='train')
    plt.plot(epochs_len, CNN_history.history['val_accuracy'], 'rx-', label='test')
    plt.title('Plot History: Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(epochs_len)
    plt.legend(['Train', 'Test'])
    plt.savefig('./images/accuracy.png')
#accuracy()

def loss():
    epochs_len = range(1, epochs + 1)
    plt.plot(epochs_len, CNN_history.history['loss'], 'bx-', label='train')
    plt.plot(epochs_len, CNN_history.history['val_loss'], 'rx-', label='test')
    plt.title('Plot History: Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(epochs_len)
    plt.legend(['Train', 'Test'])
    plt.savefig('./images/loss.png')
loss()

# Save the model
CNN_model.save('./cnn_model.keras')
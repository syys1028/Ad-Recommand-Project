import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Define a callback to stop training early if loss is less than 0.05
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.05:
            print('\n Stop training.')
            self.model.stop_training = True

# Define image size and batch size
image_size = (128, 128)
batch_size = 32

# Path to train and test directories
train_dir = os.path.join(os.getcwd(), 'train')
test_dir = os.path.join(os.getcwd(), 'test')

# Initialize ImageDataGenerators for loading and augmenting the data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training and testing data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the Sequential model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(128, 128, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(train_generator.num_classes, activation='softmax'))  # Output as per number of classes in your data
model.summary()

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with the training data
callbacks = myCallback()
history = model.fit(
    train_generator,
    epochs=5,
    batch_size=4,
    callbacks=[callbacks]
)

# Plot training accuracy and loss
plt.title('Accuracy')
plt.plot(history.history['accuracy'])
plt.show()

plt.title('Loss')
plt.plot(history.history['loss'])
plt.show()

# Evaluate the model with the testing data
loss_and_accuracy = model.evaluate(test_generator)

print('accuracy = ' + str(loss_and_accuracy[1]))
print('loss = ' + str(loss_and_accuracy[0]))

# Save the trained model
model.save('obs_model.keras')  # Save the model in .keras format

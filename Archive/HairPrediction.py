from pathlib import Path
import imghdr
import os
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

data_dir = "hair_types"
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(data_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")
            os.remove(filepath)
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
            os.remove(filepath)



# Define a function to apply sharpening to images
def sharpen_image(image):
    # Define the sharpening kernel for each channel
    kernel = tf.constant([[[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]]], dtype=tf.float32)
    # Expand the kernel to match the number of channels
    kernel = tf.tile(kernel, [1, 1, 3])
    # Apply convolution with the sharpening kernel
    sharpened_image = tf.nn.conv2d(image, tf.expand_dims(tf.transpose(kernel, perm=[2, 0, 1]), axis=-1), strides=[1, 1, 1, 1], padding='SAME')
    # Clip values to ensure they stay within valid range
    sharpened_image = tf.clip_by_value(sharpened_image, 0, 1)
    return sharpened_image

image_size = (64, 64)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "hair_types/",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    labels='inferred',
    label_mode='categorical',
    interpolation='bicubic',
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "hair_types/",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size, 
    labels='inferred',
    label_mode='categorical',
    interpolation='bicubic',
)

print(train_ds.class_names)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dropout(0.2))
model.add(keras.Input(shape=image_size + (3,))) # 64, 64, 3
model.add(layers.Rescaling(1.0 / 255))

# model.add(layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='valid', dilation_rate=1))
# model.add(layers.Activation("relu"))
# layers.MaxPool2D(pool_size=(2, 2))

model.add(layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='valid', dilation_rate=1))
model.add(layers.Activation("relu"))
layers.MaxPool2D(pool_size=(2, 2))

model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', dilation_rate=1))
model.add(layers.Activation("relu"))
layers.MaxPool2D(pool_size=(2, 2))

model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', dilation_rate=1))
model.add(layers.Activation("relu"))
layers.MaxPool2D(pool_size=(2, 2))

model.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', dilation_rate=1))
model.add(layers.Activation("relu"))
layers.MaxPool2D(pool_size=(2, 2))

model.add(layers.GlobalAveragePooling2D())
# model.add(layers.GlobalMaxPool2D())
model.add(layers.Activation("relu"))
model.add(layers.Dense(3))
# model.add(layers.Activation("softmax"))
model.add(layers.Activation("softmax"))
tf.keras.utils.plot_model(model, to_file='model_test.png', show_shapes=True)

# epochs = 1
# epochs = 75
# epochs = 25
# epochs = 16
# epochs = 18
# epochs = 40
# epochs = 35
epochs = 45
# epochs = 50
# epochs = 100
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    # optimizer=keras.optimizers.RMSprop(1e-3),
    # optimizer=keras.optimizers.SGD(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy", "precision", "categorical_accuracy"],
)

# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

# history = model.fit(train_ds, epochs=epochs, validation_data=(val_ds), callbacks=[callback])
history = model.fit(train_ds, epochs=epochs, validation_data=(val_ds))


# img = keras.preprocessing.image.load_img(
#     "hair_types/Curly_Hair/02dac897d1dec9ba8c057a11d041ada8--layered-natural-hair-natural-black-hairstyles.jpg", target_size=image_size
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create batch axis

# predictions = model.predict(img_array)
# print(
#     "This image is %.2f percent curly hair, %.2f percent straight hair, and %.2f percent wavy hair."
#     % tuple(predictions[0])
# )
print(val_ds)

# Make predictions on the validation data
predictions = model.predict(val_ds)
y_true = np.concatenate([y for x, y in val_ds], axis=0)

# Convert one-hot encoded labels to integer labels
y_true_int = np.argmax(y_true, axis=1)
y_pred = np.argmax(predictions, axis=1)

# Calculate metrics
precision = precision_score(y_true_int, y_pred, average='weighted')
recall = recall_score(y_true_int, y_pred, average='weighted')
f1 = f1_score(y_true_int, y_pred, average='weighted')
accuracy = accuracy_score(y_true_int, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



# print("========================================================")
# import matplotlib.pyplot as plt
# import numpy
# from sklearn import metrics


correct_Predictions = {'Curly_Hair': 0, 'Straight_Hair': 0, 'Wavy_Hair': 0}
Incorrect_Predictions = {'Curly_Hair': 0, 'Straight_Hair': 0, 'Wavy_Hair': 0}
all_labels = []
all_predictions = []

for hair_type in os.listdir('hair_types'):
  hair_type_dir = os.path.join(data_dir, hair_type)
  print(hair_type_dir)
  for filename in os.listdir(hair_type_dir):
    image_path = os.path.join(hair_type_dir, filename)

    # Load and preprocess image
    img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Make prediction and convert probabilities to class label
    predictions = model.predict(img_array)
    predicted_class = tf.math.argmax(predictions[0]).numpy()
    # predicted_label = list(model.class_indices.keys())[predicted_class]
    if predicted_class == 0:
        predicted_label = 'Curly_Hair'
    elif predicted_class == 1:
        predicted_label = 'Straight_Hair'
    elif predicted_class == 2:
        predicted_label = 'Wavy_Hair'
    else:
        print("Error")
        exit(1)
    print(str(predicted_class) + " | " + str(predictions) + " Class: " + predicted_label + " vs " + hair_type)
    # predicted_label = val_ds.class_names[predicted_class]
    # print(predictions)
    # print(predicted_label + " is " + " " + " | " + hair_type)
    # Update true/false positives based on ground truth (hair_type)
    if predicted_label == hair_type:
      correct_Predictions[hair_type] += 1
    elif predicted_label != hair_type:
      Incorrect_Predictions[hair_type] += 1

    
    all_labels.append(hair_type)
    all_predictions.append(predicted_label)


Curly_count = 0
for root_dir, cur_dir, files in os.walk(r'hair_types/Curly_Hair'):
    Curly_count += len(files)
print('Curly count:', Curly_count)


Straight_count = 0
for root_dir, cur_dir, files in os.walk(r'hair_types/Straight_Hair'):
    Straight_count += len(files)
print('Stright count:', Straight_count)

Wavy_count = 0
for root_dir, cur_dir, files in os.walk(r'hair_types/Wavy_Hair'):
    Wavy_count += len(files)
print('Wavy count:', Wavy_count)

# print("========================================================")
# # Calculate overall accuracy (optional)
# total_images = sum(true_positives.values())
# accuracy = sum(true_positives.values()) / sum([Curly_count, Straight_count, Wavy_count])
# print(f"Overall accuracy: {accuracy:.4f}")

# #Calculate overall precision (Optional)
# total_images = sum(true_positives.values())
# precision = sum(true_positives.values()) / (sum(true_positives.values()) + sum(false_positives.values()))
# print(f"Overall precision: {precision:.4f}")

# # Calculate overall recall (optional)
# total_images = sum(true_positives.values())
# recall = sum(true_positives.values()) / (sum(true_positives.values()) + sum(false_negatives.values()))
# print(f"Overall recall: {recall:.4f}")

# # Calculate overall F1 score (optional)
# total_images = sum(true_positives.values())
# f1 = 2 * (precision * recall) / (precision + recall)
# print(f"Overall F1 score: {f1:.4f}")

# # Calculate confusion matrix (optional)
# actual = numpy.random.binomial(1,.9,size = 1000)
# predicted = numpy.random.binomial(1,.9,size = 1000)

# confusion_matrix_results = metrics.confusion_matrix(all_labels, all_predictions)
# print("Confusion Matrix:\n", confusion_matrix_results)

print("========================================================")
print("Correct:")
for hair_type, count in correct_Predictions.items():
  print(f"  {hair_type}: {count}\n")

print("\nIncorrect:")
for hair_type, count in Incorrect_Predictions.items():
  print(f"  {hair_type}: {count}\n")


print("========================================================")


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

import matplotlib.pyplot as plt

def plot_history(history):
    # Plotting the training history

    plt.figure(figsize=(12, 5))
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot precision
    plt.figure()
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
plot_history(history)
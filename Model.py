import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Enable Apple Metal Acceleration
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Directories
data_dir = "/Users/dhruvgadhavi/Documents/archive/"
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")

# Image Parameters
img_size = (224, 224)
batch_size = 32

# Data Loading and Preprocessing using image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical' # Use categorical for multi-class
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical' # Use categorical for multi-class
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',  # Use categorical for multi-class
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False  # Important for evaluation
)

# Define Model
base_model = keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False  # Freeze base model weights

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_ds.class_names), activation='softmax')  # Output layer based on the number of classes
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping Callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Model Training
epochs = 20 # Or more!

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping]
)
exit() # ADD THIS LINE

# Evaluate the Model
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Save Model
print("Saving model in Keras format (.keras)") # Add this line
model.save("brain_tumor_detector.keras") # Or "brain_tumor_detector"


# Data Visualization (Training History)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# Function for Prediction and Evaluation with Recommendations
def predict_tumor(image_path):
    """
    Predicts the presence of a brain tumor in an image and provides recommendations.
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, img_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0]) # Index of the most probable class
        class_names = ['no_tumor', 'glioma', 'meningioma', 'pituitary'] # Classes must match folder names!
        predicted_class = class_names[class_index]
        confidence = predictions[0][class_index]  # Probability of the predicted class

        recommendations = ""
        if predicted_class == "no_tumor":
            recommendations = "No tumor detected. Maintain a healthy lifestyle."
        else:
            recommendations = f"A {predicted_class} tumor is detected. "
            recommendations += "Consult a neurologist for further evaluation and treatment options. "
            recommendations += "Follow a balanced diet and consider consulting with a nutritionist."

        return f"Prediction: {predicted_class} (Confidence: {confidence:.4f})\n{recommendations}"

    except Exception as e:
        return f"Error processing image: {e}"

# Example Usage
image_path = "/Users/dhruvgadhavi/Documents/archive/Testing/glioma/image(1).jpg"  # Replace with the path to your image
result = predict_tumor(image_path)
print(result)

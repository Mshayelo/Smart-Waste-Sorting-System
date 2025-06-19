import os             # For handling file paths and directories
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns           # For more advanced and styled plots
from tensorflow.keras.preprocessing.image import ImageDataGenerator    # For image loading and preprocessing


# Load the pre-trained model saved from training
model = tf.keras.models.load_model('Exercise 3 Assessment/Model/garbage_classifier.h5')

#Load class names from the text file
# These names help interpret which class index corresponds to which label

with open('Exercise 3 Assessment/Model/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]         # Read each line, remove newline characters

# Set the path to the test dataset directory
test_dir = 'Exercise 3 Assessment/Split_Dataset/test'

# Define preprocessing parameters
# Get input size from the loaded model to ensure compatibility (e.g., 256x256)

IMG_SIZE = model.input_shape[1:3]
batch_size = 32

#Create a test data generator
# Rescale the pixel values (normalize) but do not apply augmentation to keep test data clean

datagen = ImageDataGenerator(rescale=1./255)

# Load images from the test directory using the same preprocessing and class mode as in training
# Shuffle is set to False so we can match predictions with actual labels correctly

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

#Predict the test dataset using the loaded model
# The model returns probabilities for each class, and we take the class with the highest score

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)  # Get index of highest probability (predicted class)
y_true = test_data.classes               # True class labels from the data generator

# Generate and print a classification report
#This shows precision, recall, f1-score, and accuracy for each class

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# Create and display a confusion matrix
# A confusion matrix helps visualize how often the model predicts each class correctly vs. incorrectly

cm = confusion_matrix(y_true, y_pred)  # Compute the matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt='d', cmap='Blues')  # Blue color map
plt.xlabel('Predicted')                # Label for predicted classes
plt.ylabel('True Label')              # Label for actual classes
plt.title('Confusion Matrix')         # Plot title
plt.show()                            # Display the plot



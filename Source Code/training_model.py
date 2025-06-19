import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# # Define paths to the training and validation datasets and where to save the model and class names

train_dir = 'Exercise 3 Assessment/Split_Dataset/train'
val_dir = 'Exercise 3 Assessment/Split_Dataset/val'
model_save_path = 'Exercise 3 Assessment/Model/garbage_classifier.h5'       # Path where the final model will be saved in HDF5 file format
class_names_path = 'Exercise 3 Assessment/Model/class_names.txt'         # File to save class labels for later use

# Set constant parameters for the image size, batch size, and number of training epochs
IMG_SIZE = (256, 256)  # Image size expected by MobileNetV2
BATCH_SIZE = 32        # Number of images processed together during training
EPOCHS = 20            # Number of full training cycles over the dataset

# Create a data generator for training with real-time data augmentation
# This helps the model generalize better by slightly modifying the images each time (e.g., flipping, rotating)


train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values (0-255) to (0-1)
    rotation_range=20,            # Randomly rotate images by up to 20 degrees
    zoom_range=0.2,               # Random zoom within images
    width_shift_range=0.1,        # Random horizontal shift
    height_shift_range=0.1,       # Random vertical shift
    horizontal_flip=True          # Randomly flip images horizontally
)

# Create a data generator for validation with only normalization (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training images from the directory

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,               # Resize all images to the specified size
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load and preprocess validation images

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# Save class names for future use like making predictions or displaying results

os.makedirs(os.path.dirname(class_names_path), exist_ok=True)
with open(class_names_path, 'w') as f:
    for class_name in train_generator.class_indices.keys():
        f.write(f"{class_name}\n")

# Load MobileNetV2 model as the base (pretrained on ImageNet), without the final classification layer (without top layer)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),                # Add 3 for RGB color channels
    include_top=False,                          # Do not include the original final layer (we'll add our own)
    weights='imagenet'
)

#Freeze the base model so it doesn't get updated during the first training phase
base_model.trainable = False

## Build the final model by adding new layers (classification head) on top of the base model
model = models.Sequential([
    base_model,                         # Using the pre-trained MobileNetV2 as the base
    layers.GlobalAveragePooling2D(),    # Reduces dimensions while keeping important information
    layers.Dense(128, activation='relu'),   # Fully connected layer with ReLU activation
    layers.Dropout(0.3),                            # Dropout to prevent overfitting
    layers.Dense(len(train_generator.class_indices), activation='softmax')    # Output layer with softmax for multiple classes
])

#  Compile the model with Adam optimizer, a suitable loss function, and accuracy as a metric

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),       # Slightly higher learning rate for frozen base
              loss='categorical_crossentropy',                              #Quite suitable for multi-class problems
              metrics=['accuracy'])

# Train the model according to the number of times stored within the Epochs Parameter
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)


# Fine-tune top layers of base model for better accuracy
# Unfreeze the base model so we can fine-tune it (let it learn from my specific dataset)

base_model.trainable = True
# # Recompile the model with a lower learning rate to avoid making large updates to the pre-trained weights

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),    # Lower learning rate for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training (fine-tuning) the full model for a few more epochs to improve performance
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10               # Fine-tuning the model with just fewer epochs
)

# Save the final trained model as an HDF5 file in the specified file path
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

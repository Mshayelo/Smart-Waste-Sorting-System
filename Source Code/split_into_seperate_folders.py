import os              # For handling file paths and directories
import shutil           # Used to copy files between folders
import random               # Used to shuffle image file lists randomly


# Dataset Preparation

# Define original dataset directory and the folder where the split dataset will be saved
data_dir = 'Exercise 3 Assessment/My_Dataset/Garbage classification/Garbage classification'
output_dir = 'Exercise 3 Assessment/Split_Dataset'

# Get class names(folder names) from the original dataset directory
#Using a List Comprehension to simplify
# Only include folders (not files) in the list

class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# # Define the split ratios for training, validation, and testing
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create folder structure for split dataset
# For each split (train/val/test) and each class, create corresponding folders

for split in ['train', 'val', 'test']:
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# Function to split and copy files
# Split and copy files to new folders
# For each class, shuffle its image files and divide them into train/val/test

for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)  # Path to current class folder
    images = os.listdir(class_path)                  # List of all images in the class

    random.shuffle(images)                           # Randomize the order of images

    total_images = len(images)                       # Total number of images in the class
    train_end = int(train_ratio * total_images)      # Index where training data ends
    val_end = train_end + int(val_ratio * total_images)  # Index where validation data ends

    # Slice the list of images into three subsets
    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    #Copy the images to their corresponding new folders
    for file in train_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(output_dir, 'train', class_name, file))
    for file in val_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(output_dir, 'val', class_name, file))
    for file in test_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(output_dir, 'test', class_name, file))

# Final confirmation message
print("✅ Images split into train, val, test folders successfully. Quality preserved!")

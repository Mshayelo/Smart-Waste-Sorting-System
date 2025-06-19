import pickle                       # For saving and loading Python objects
import matplotlib.pyplot as plt
from training_model import history       # Import the training history object from training_model.py


# Save the training history to a file
# The training history contains metrics such as accuracy and loss per epoch

with open('Exercise 3 Assessment/Model/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)       # Save the dictionary (history.history) in binary format

# Load the saved training history from the file
#This step can be used in a separate session or script to reuse the stored training performance

with open('Exercise 3 Assessment/Model/training_history.pkl', 'rb') as f:
    history = pickle.load(f)                            # Load the history dictionary

#Plot The Accuracy Graph

plt.figure(figsize=(12, 5))             # Set the overall figure size

plt.subplot(1, 2, 1)                # Create the first subplot (1 row, 2 columns, position 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)                       # Add gridlines for easier reading

# Plot Loss Graph
plt.subplot(1, 2, 2)                # Create the second subplot (1 row, 2 columns, position 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)                  # Add gridlines

# Final Layout and display
plt.tight_layout()                       # Adjust spacing to prevent overlap
plt.show()                               # Display both plots


import cv2
import numpy as np
import tensorflow as tf


# Load the trained model
# This is the pre-trained garbage classification model that I trained earlier.
model_path = 'Exercise 3 Assessment/Model/garbage_classifier.h5'
model = tf.keras.models.load_model(model_path)

#Load class names for predictions
# These are the categories the model can predict (like paper, plastic, etc.).
# The names are read from a text file and stored in a list.

with open('Exercise 3 Assessment/Model/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"Loaded Classes: {class_names}")  # Printing the loaded class names for confirmation

#Setting prediction parameters
CONFIDENCE_THRESHOLD = 0.7  # If confidence is lower than this, we won’t trust the result
IMG_SIZE = model.input_shape[1:3]  # Input image size expected by the model, e.g., (224, 224)

# Loading face detection model (Haar Cascade) , (Optional safety check)
# This is used as a safety check. If a face is detected, the system avoids making a prediction.

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Opening the webcam
# Start webcam capture (0 = default camera)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a single frame from the webcam
    if not ret:
        break               # Exit the loop if the camera frame cannot be read

    height, width, _ = frame.shape   # Get height and width of the video frame

    #Defining the Region of Interest (ROI)
    # ROI is the area in the middle of the screen where the user must place the object.
    # I reduced the margin so that the box is larger (bigger detection window).

    roi_margin = 0.25  # 25% margin on each side of the frame
    top_left = (int(width * roi_margin), int(height * roi_margin))
    bottom_right = (int(width * (1 - roi_margin)), int(height * (1 - roi_margin)))

    #  Blur the entire frame to focus user attention on the ROI(the centered box)
    blurred_frame = cv2.GaussianBlur(frame, (55, 55), 0)

    # Copy the clear ROI (original image) into the blurred background
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    blurred_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = roi

    # Drawing a green rectangle to indicate where to place the object for detection
    cv2.rectangle(blurred_frame, top_left, bottom_right, (0, 255, 0), 2)

    # Convert the ROI to grayscale and detect any faces
    # If a face is detected, we won’t make a prediction (for privacy and safety).

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5)

    # Preprocess the ROI image before sending it to the model
    img = cv2.resize(roi, IMG_SIZE)         # Resizing the ROI to match model input size
    img = img / 255.0                       # Normalizing pixel values to range between 0 and 1
    img = np.expand_dims(img, axis=0)   # Adding a batch dimension: shape becomes (1, H, W, 3)

    #  Predict the class of the object in the ROI(Centered frame)

    predictions = model.predict(img, verbose=0)   # Run the image through the model
    confidence = float(np.max(predictions))         # Get the highest confidence score
    predicted_class = class_names[int(np.argmax(predictions))]          # Find the class with the highest score

    # Deciding what text to display based on the result
    # If a face is found in the image, it does not predict to avoid errors

    if len(faces) > 0:
        display_text = "Unknown (Face detected)"
    elif confidence < CONFIDENCE_THRESHOLD:                 # If the model is not confident enough, we don’t trust the prediction
        display_text = f"Unknown ({confidence*100:.2f}%)"
    else:
        display_text = f"{predicted_class} ({confidence*100:.2f}%)"        # Otherwise, show the predicted class with its confidence

    # Show the prediction on the screen above the green box
    cv2.putText(blurred_frame, display_text, (top_left[0], top_left[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #Show the final result window with all overlays

    cv2.imshow('Smart Waste Sorting - Real-Time', blurred_frame)

    # Press the "q" key to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all windows after the loop ends
cap.release()
cv2.destroyAllWindows()

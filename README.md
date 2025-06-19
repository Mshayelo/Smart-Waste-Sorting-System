# Smart-Waste-Sorting-System 

An intelligent vision-based waste classification system that identifies and classifies recyclable waste into five categories: **plastic**, **paper**, **metal**, **glass**, and **cardboard**. This project uses **Convolutional Neural Networks (CNN)**, **MobileNetV2**, and **Transfer Learning** to create a lightweight, high-performance image classifier suitable for real-time deployment.


##  Project Objectives

- Develop a machine learning model capable of classifying images of recyclable waste.
- Deploy a webcam-based real-time detection system.
- Promote sustainability through smarter waste management.


##  Features
-  Built using MobileNetV2 with Transfer Learning
-  Real-time classification via webcam using OpenCV
-  Visualization of training and evaluation metrics
-  Automatic dataset splitting (Train/Val/Test)
-  Confusion matrix and classification report generation

##  Setup & Installation

1. **Clone this repository:**

 git clone https://github.com/your-username/smart-waste-sorting-system.git  
 cd smart-waste-sorting-system

# Install required dependencies:

pip install -r requirements.txt

# Prepare your dataset:

Place your dataset inside My_Dataset/Garbage classification/ (5 folders: plastic, paper, metal, glass, cardboard).

Run split_dataset.py to generate train/val/test folders automatically.

# Model Training
To train the model and save it as garbage_classifier.h5:

python training_model.py

Fine-tuning is handled inside the same script after initial training.

 # Model Evaluation
To view performance metrics, confusion matrix, and classification report:

python model_evaluation.py and  python model_perfomance.py

Graphs (accuracy/loss) will also be shown using matplotlib.

 # Real-Time Classification
To use the webcam classification system:

python real_time_detection.py
Place the object inside the green box.

Avoid placing faces in view (the system blurs background and skips prediction if a face is detected).

# Built With
Python

TensorFlow & Keras

OpenCV

Matplotlib & Seaborn

Scikit-learn

# Skills Demonstrated
Transfer Learning with MobileNetV2

Data Augmentation & Preprocessing

Real-Time Object Detection

Performance Evaluation (Precision, Recall, F1)

Ethical considerations in ML (privacy, generalizability)

Smart-Waste-Sorting-System

An intelligent vision-based waste classification system that identifies and classifies recyclable waste into five categories: **plastic**, **paper**, **metal**, **glass**, and **cardboard**.  

This project uses **Convolutional Neural Networks (CNN)**, **MobileNetV2**, and **Transfer Learning** to build a lightweight and effective classifier for real-time deployment.

> This project forms part of my **Information Systems 315** coursework at the **University of the Western Cape**.  
> For an in-depth explanation of the problem, solution approach, evaluation, and future recommendations, see:  
 **[Project_Report.pdf](./Project_Report.pdf)**


##  Project Objectives

- Build a CNN model to classify waste images into 5 recyclable categories
- Leverage MobileNetV2 and Transfer Learning for efficient performance
- Enable **real-time object classification** through webcam
- Support sustainable development by improving waste management processes


##  Features

-  Transfer Learning with MobileNetV2
-  Real-time classification via webcam (OpenCV)
-  Automatic dataset splitting into train/val/test
-  Visual training curves and evaluation metrics
-  Classification report & Confusion matrix visualization
-  Face detection safety layer (blocks classification when human face is detected)



## Preparing Dataset

This system is trained on the **TrashNet Dataset** — a public dataset for garbage classification.

-  Contains labeled images for: `cardboard`, `glass`, `metal`, `paper`, and `plastic`

#### Download Instructions:

1. Download from the official source:  
   TrashNet Dataset (by Gary Thung) (https://github.com/garythung/trashnet)

2. Place the extracted folders like this:

My_Dataset/
└── Garbage classification/
├── cardboard/
├── glass/
├── metal/
├── paper/
└── plastic/


3. Run the script below to automatically split the dataset:

#### python split_dataset.py

# Setup & Installation

### Clone the repository:

git clone https://github.com/your-username/smart-waste-sorting-system.git

cd smart-waste-sorting-system

### Install dependencies:

#### pip install -r requirements.txt

### Model Training
To train the model (including fine-tuning) and save it:

#### python training_model.py

Trained model will be saved to:
#### Exercise 3 Assessment/Model/garbage_classifier.h5

Class names saved in:
#### class_names.txt

### Model Evaluation
To view classification report and confusion matrix:

 #### python model_evaluation.py

Outputs include accuracy, precision, recall, and F1-score per class

A heatmap visual of the confusion matrix will be displayed

### Real-Time Classification (Webcam)
To launch the real-time waste classification interface:

 #### python real_time_detection.py

A green bounding box is displayed — place your object inside it

Background is blurred for focus; face detection is used to block predictions if a human face is detected

### Training Metrics Visualization
To visualize the training and validation accuracy/loss:

python model_perfomance.py
This will load the saved training history and plot performance graphs over the epochs.

# Skills Demonstrated
 Transfer Learning & MobileNetV2

 Data Augmentation & Image Preprocessing

 Real-time Computer Vision using OpenCV

 Model Evaluation & Metric Reporting

 Ethical ML Practices: Privacy-Aware Design

# License
This project is for academic use and knowledge-sharing purposes.
Please credit the original TrashNet dataset creators and cite this repository if reused.

# Additional Info
A detailed breakdown of the methodology, architecture, evaluation, and learnings can be found in:
  **[Project_Report.pdf](./Project_Report.pdf)**

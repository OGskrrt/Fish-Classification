# Fish Species Classification Using Deep Learning

##[Kaggle Notebook Link]([https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset/data](https://www.kaggle.com/code/ogskrrrt/fish-classification)) 

## Overview
This project focuses on the classification of nine different seafood types using a large-scale dataset collected from a supermarket in Izmir, Turkey. The dataset contains 9000 images of various fish species and shrimp, which are labeled into categories for classification tasks. The goal of this project is to develop a robust image classification model using a deep learning approach, specifically a multi-layer Artificial Neural Network (ANN), to accurately classify the seafood images.

## Dataset
The dataset consists of images of nine seafood types, collected for the purpose of segmentation, feature extraction, and classification. The dataset contains 1000 augmented images per class, including their corresponding ground truth labels. The seafood classes are:

1. Gilt-Head Bream
2. Red Sea Bream
3. Sea Bass
4. Red Mullet
5. Hourse Mackerel
6. Black Sea Sprat
7. Striped Red Mullet
8. Trout
9. Shrimp

Each image is provided in `.png` format, and the ground truth labels are stored in separate directories. The dataset contains a total of 9000 images, with an equal number of images in each class.

## Project Workflow

### 1. **Data Understanding**
The first step was to analyze the dataset by exploring the structure and characteristics of the data, including the image file paths, class labels, and dimensions. Data analysis included checking the distribution of classes, examining image shapes, and determining pixel value ranges for normalization.

- **Image resolution**: All images were found to have consistent dimensions of 445x590 pixels and three color channels.
- **Class distribution**: The dataset was perfectly balanced, with each class contributing 11.11% of the total dataset.

### 2. **Data Preparation**
To prepare the data for training, several preprocessing steps were performed:
- **Image Augmentation**: An `ImageDataGenerator` was used to generate augmented images in real-time by applying transformations such as scaling, zooming, rotation, and shifting.
- **Rescaling**: All images were rescaled by dividing pixel values by 255 to normalize them between 0 and 1.
- **Training and Validation Split**: The dataset was split into a training set (90%) and a validation set (10%).

### 3. **Modeling**
The core of the project involved building and training a deep learning model using TensorFlow and Keras. A multi-layer **Artificial Neural Network (ANN)** was built with the following architecture:

- **Flatten Layer**: Converts the 2D image arrays into 1D vectors.
- **Dense Layers**: Fully connected layers with 1024, 512, 256, and 128 neurons, each followed by a Batch Normalization and Dropout layer for regularization.
- **Output Layer**: A softmax layer with 9 units for multi-class classification.

The model was compiled using the Adam optimizer with a learning rate of 0.001 and categorical cross-entropy as the loss function. Accuracy, Precision, and Recall were used as evaluation metrics.

### 4. **Model Training**
The model was trained over 50 epochs using early stopping and model checkpointing to prevent overfitting. The training process showed a steady improvement in both training and validation accuracy.

- **Training Set**: 16,200 images.
- **Validation Set**: 1,800 images.

### 5. **Evaluation**
After training, the model achieved a final test accuracy of **79.5%**. The performance was evaluated using accuracy, precision, recall, and loss on both the training and validation sets. The final model performed well with acceptable accuracy and recall values, though there is room for further improvement.

- **Test Loss**: 0.63
- **Test Accuracy**: 79.5%
- **Precision**: 82.8%
- **Recall**: 75.6%

### 6. **Results Visualization**
Two key graphs were plotted to visualize the training process:
- **Accuracy Plot**: Shows how the model's accuracy improved over the epochs.
- **Loss Plot**: Shows the training and validation loss over time, helping identify overfitting patterns.

## Conclusion
This project successfully applied deep learning techniques to classify seafood images into nine distinct categories. The model achieved good performance and demonstrated the ability to generalize well to unseen data. 

### Future Work
- **Model Enhancement**: Explore more complex architectures such as Convolutional Neural Networks (CNNs) for potentially better performance.
- **Fine-Tuning**: Perform hyperparameter optimization to further improve the model's accuracy and reduce loss.
- **Deployment**: Convert the trained model into a web application for real-time fish classification.

## Installation and Setup
To reproduce the results of this project, follow these steps:
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt   
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset/data) and place it in the specified directory.
4. Run the notebook for training the model.

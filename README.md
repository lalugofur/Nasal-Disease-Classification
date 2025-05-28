# Nasal-Disease-Classification-Using-Deep-Learning

## Project Overview

This project focuses on classifying nasal disease images using deep learning techniques. The main objective is to develop and compare convolutional neural network (CNN) architectures for accurately recognizing and classifying visual patterns in medical images of nasal diseases.

## Dataset and Preprocessing

The dataset is split into three subsets to ensure objective evaluation of the model's performance:

- **Training set:** 80%
- **Validation set:** 10%
- **Test set:** 10%

Each subset is stored in separate directories. Images are resized to **224 x 224 pixels** to match the input requirements of the CNN architectures used.

## Training Parameters

Key training parameters include:

- **Batch size:** 32
- **Number of epochs:** 50
- **Input image size:** 224 x 224 pixels

These parameters are carefully selected to balance training stability, computational efficiency, and optimal model performance.

## Data Augmentation

To enhance model generalization and reduce overfitting, data augmentation is applied on the training data using various techniques, including:

- Rotation (up to 30 degrees)
- Horizontal and vertical shifting
- Zooming
- Shearing
- Horizontal flipping

This augmentation strategy increases data variability by simulating realistic transformations of the nasal images.

## Model Architectures

The project evaluates the performance of two state-of-the-art CNN architectures:

- **MobileNetV2**
- **EfficientNetV2B3**

Both architectures are fine-tuned on the augmented dataset to improve their ability to classify complex visual features in nasal disease images.

## Objectives

- Compare classification accuracy and other evaluation metrics between MobileNetV2 and EfficientNetV2B3.
- Assess the impact of data augmentation on model performance.
- Develop a reliable deep learning pipeline for nasal disease classification to support medical diagnosis.

## Usage

The repository contains scripts for:

- Data preprocessing and augmentation
- Model training and evaluation
- Visualization of training history and performance metrics

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- PIL (Pillow)

Install dependencies via:

```bash
pip install -r requirements.txt

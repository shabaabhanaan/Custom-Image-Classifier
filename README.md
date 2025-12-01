# Custom Image Classifier

A simple **Convolutional Neural Network (CNN)** model built with **TensorFlow** and deployed using **Streamlit**. This project allows users to upload an image and classify it into predefined categories.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Details](#model-details)
- [License](#license)

---

## Overview

This project demonstrates how to build a CNN to classify images into multiple categories. The model is trained on a custom dataset and then deployed as an interactive web app using Streamlit.

---

## Features

- Train a CNN on a custom image dataset
- Upload images via a web interface
- Classify images and display confidence scores
- Supports PNG and JPG image formats

---

## Requirements

- Python 3.8+
- TensorFlow
- Streamlit
- Pillow (PIL)
- NumPy

Install the dependencies using:

```bash
pip install tensorflow streamlit pillow numpy
.
├── dataset/
│   ├── train/       # Training images organized by class folders
│   └── test/        # Test images organized by class folders
├── custom_cnn_model.h5  # Trained model (generated after training)
├── app.py           # Streamlit app to classify images
└── README.md

Usage

Clone the repository:

git clone https://github.com/your-username/custom-image-classifier.git
cd custom-image-classifier


Run the Streamlit app:

streamlit run app.py


Upload an image in the web app to classify it. The app will display the predicted class and confidence percentage.

Model Details

Input: 64x64 RGB images

Architecture:

Rescaling layer to normalize pixel values

3 Convolutional layers with ReLU activation

MaxPooling layers to reduce spatial dimensions

Fully connected Dense layer with 128 neurons

Output layer with softmax activation for multi-class classification

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Epochs: 10

Batch Size: 16

License

This project is licensed under the MIT License.

Author: Shabaab Hanaan

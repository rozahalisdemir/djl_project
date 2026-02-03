# Handwritten Digit Recognition with DJL 

This project is a Deep Learning application built with **Java** using the **Deep Java Library (DJL)**. It implements a Multilayer Perceptron (MLP) neural network to recognize handwritten digits from the MNIST dataset.

The system is capable of training a model from scratch, saving it, and running inference on real-world images with high accuracy.

## 🚀 Key Features

* **Training Pipeline:** Trains an MLP model using the MNIST dataset (60,000 training images).
* **Model Architecture:**
    * Input Layer: 784 nodes (28x28 pixels flattened)
    * Hidden Layer 1: 128 nodes (ReLU activation)
    * Hidden Layer 2: 64 nodes (ReLU activation)
    * Output Layer: 10 nodes (Softmax activation for digits 0-9)
* **Inference:** Loads the trained model to predict digits from external image URLs.
* **Performance:** Achieves ~99% accuracy on the test set.

## 🛠️ Tech Stack

* **Language:** Java 11+
* **Framework:** Deep Java Library (DJL)
* **Engine:** PyTorch (via DJL engine)
* **Build Tool:** Maven

## 📂 Project Structure

```bash
src/main/java/djl_project/
├── HelloDJL.java   # Main class for Training. Downloads MNIST, trains the model, and saves it.
└── Inference.java  # Main class for Testing. Loads the saved model and predicts a digit from an image URL.

# feedforward-neural-networks-matlab
![MATLAB](https://img.shields.io/badge/MATLAB-R2018b%2B-orange?logo=mathworks)
![Neural Networks](https://img.shields.io/badge/AI-Feedforward%20NN-blue)
![Status](https://img.shields.io/badge/Status-Academic%20Project-green)

## 📋 Overview

This project implements a computer vision system capable of identifying and classifying geometric shapes using **Feedforward Neural Networks** in MATLAB.

The system processes raw images, applies preprocessing techniques, and feeds them into a neural network to classify six distinct geometric shapes: **Circle, Kite, Parallelogram, Square, Trapezoid, and Triangle**. The project explores various network architectures, training algorithms, and activation functions to optimize classification accuracy.

## 🎯 Objectives

* **Implementation:** Develop a complete pipeline for training feedforward neural networks for shape recognition.
* **Exploration:** Experiment with different network topologies, activation functions, and training algorithms.
* **Evaluation:** Assess model performance across multiple datasets.
* **Comparison:** Analyze results across multiple network architectures to find the best model.

## 📁 Project Structure

```text
TPCR2425/
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
│
├── docs/                        # Project Documentation
│   ├── RelatorioCR.docx         # Final Report
│   └── TP2_IIA_2425.pdf         # Assignment Brief
│
├── scripts/                     # MATLAB Source Code
│   ├── convertImage_*.m         # Image Preprocessing & Loading functions
│   │   ├── convertImage_all.m   # Loads all datasets (start, train, test)
│   │   ├── convertImage_start.m # Loads 'start' dataset
│   │   ├── convertImage_train.m # Loads 'train' dataset
│   │   ├── convertImage_test.m  # Loads 'test' dataset
│   │   └── convertImage_draw.m  # Loads 'draw' dataset
│   │
│   ├── train_*.m                # Network Training Scripts
│   │   ├── train_start_a.m      # Part A: Initial training (start dataset)
│   │   ├── train_train_b.m      # Part B: Full training (train dataset)
│   │   ├── train_test_c_i.m     # Part C-I: Evaluation on test set
│   │   ├── train_test_c_ii.m    # Part C-II: Training with test data
│   │   └── train_all_c_iii.m    # Part C-III: Training on combined data
│   │
│   ├── test_draw_d.m            # Part D: Evaluation on hand-drawn images
│   └── app.mlapp                # Interactive MATLAB GUI App
│
├── models/                      # Pre-trained Neural Network Models
│   ├── rede_top_73_162459.mat   # 73% Accuracy
│   ├── rede_top_64_152746.mat   # 64% Accuracy
│   └── rede_top_62_162036.mat   # 62% Accuracy
│
├── draw/                        # Hand-drawn dataset
├── start/                       # Initial dataset (5 per class)
├── train/                       # Training dataset (50 per class)
└── test/                        # Testing dataset (10 per class)
```
## 🔧 Prerequisites

To run this project, you need:
* **MATLAB R2018b** or higher.
* **Deep Learning Toolbox** (formerly Neural Network Toolbox).
* **Image Processing Toolbox**.

## 📊 Datasets

The project utilizes four image datasets:

| Dataset | Count (Total) | Description |
| :--- | :--- | :--- |
| **start/** | 30 images | 5 per class. Used for initial training and quick prototyping. |
| **train/** | 300 images | 50 per class. Main dataset for supervised learning. |
| **test/** | 60 images | 10 per class. Used for final evaluation. |
| **draw/** | 30 images | 5 per class. Hand-drawn images for robustness testing. |

**Target Classes:**
`Circle` • `Kite` • `Parallelogram` • `Square` • `Trapezoid` • `Triangle`

## 🚀 How to Use

### 1. Data Preparation
Images are automatically processed by the `convertImage_*.m` functions.
* **Preprocessing:** Grayscale conversion -> Resize to 25x25 -> Binarization -> Flattening.

### 2. Training Neural Networks

**Part A - Initial Training (Start)**
```matlab
cd scripts
train_start_a

Sem problema. Aqui tens o restante do conteúdo, começando exatamente na Part B até ao final do documento.

Podes copiar este bloco e colar logo após a secção "Part A" que já tens.

Markdown

**Part B - Full Training (Train)**
```matlab
cd scripts
train_train_b

**Part C - Testing & Advanced Evaluation
```cd scripts
train_test_c_i      
train_test_c_ii     
train_all_c_iii

**Part D - Hand-Drawn Test
```cd scripts
test_draw_d

### 3. GUI Application
Run `scripts/app.mlapp` in MATLAB App Designer for an interactive graphical interface.

> **Note:** Ensure you execute scripts from the `scripts/` folder or add it to your MATLAB path.

## 📈 Results

Three pre-trained models are included in the `models/` directory:

* 🥇 **Model 1:** (`rede_top_73_162459.mat`) - **73% Accuracy**
* 🥈 **Model 2:** (`rede_top_64_152746.mat`) - **64% Accuracy**
* 🥉 **Model 3:** (`rede_top_62_162036.mat`) - **62% Accuracy**

## 🔍 Image Processing Pipeline

All images undergo the following transformation pipeline:
1.  **Read:** Load PNG image.
2.  **Grayscale:** Convert to grayscale (if RGB).
3.  **Resize:** Downsample to **25x25 pixels**.
4.  **Binarize:** Convert to binary (black/white) based on threshold.
5.  **Vectorize:** Flatten 25x25 matrix into a **625-element column vector**.

## 📝 Key Functions

**`convertImage_start()`**
* Loads/processes `start/` folder.
* **Returns:** `[inputs, targets]`
    * `inputs`: 625x30 Matrix
    * `targets`: 6x30 Matrix (One-hot encoding)

**`convertImage_train()`**
* Loads/processes `train/` folder.
* **Returns:** `[inputs, targets]`
    * `inputs`: 625x300 Matrix
    * `targets`: 6x300 Matrix

**`convertImage_test()`**
* Loads/processes `test/` folder.
* **Returns:** `[inputs, targets]`
    * `inputs`: 625x60 Matrix
    * `targets`: 6x60 Matrix

**`convertImage_draw()`**
* Loads/processes `draw/` folder.
* **Returns:** `[inputs, targets]`
    * `inputs`: 625x30 Matrix
    * `targets`: 6x30 Matrix

**`convertImage_all()`**
* Loads images from start, train, and test folders.
* **Returns:** `[inputs, targets]` (Total: 390 images).

## 🛠️ Neural Network Configuration

**Topologies Tested**
* Single Layer: `[10]`, `[20]`, `[30]`
* Multi-Layer: `[20 20]`, `[20 20 20]`, `[30 20 10]`

**Activation Functions**
* `tansig` (Hyperbolic Tangent) - Standard for hidden layers.
* `logsig` (Log-Sigmoid).
* `purelin` (Linear) - Used in output layer.

**Training Functions**
* `trainlm` (Levenberg-Marquardt) - Default, fast but memory intensive.
* `traingd` (Gradient Descent).
* `traingda` (Adaptive Gradient Descent).

## 📊 Evaluation Metrics

The project utilizes:
* **Confusion Matrices** for detailed class breakdown.
* **Global Accuracy** per dataset.
* Comparative analysis between architectures.

## 📄 License & Author

**Author:** Developed for the Robotic Computing (TPCR) Practical Work 2024/2025.
**License:** Academic Project.

**References:**
* MATLAB Neural Network Toolbox Documentation
* MATLAB Image Processing Toolbox Documentation

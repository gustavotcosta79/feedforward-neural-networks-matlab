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

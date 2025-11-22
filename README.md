# feedforward-neural-networks-matlab
📋 OverviewThis project implements a computer vision system capable of identifying and classifying geometric shapes using Feedforward Neural Networks in MATLAB.The system processes raw images, applies preprocessing techniques, and feeds them into a neural network to classify six distinct geometric shapes: Circle, Kite, Parallelogram, Square, Trapezoid, and Triangle. The project explores various network architectures, training algorithms, and activation functions to optimize classification accuracy.🎯 ObjectivesImplementation: Develop a complete pipeline for training feedforward neural networks for shape recognition.Exploration: Experiment with different network topologies, activation functions (e.g., tansig, logsig), and training algorithms.Evaluation: Assess model performance across multiple datasets, including computer-generated and hand-drawn shapes.Optimization: Compare architectures to maximize generalization and accuracy.📁 Project StructurePlaintextTPCR2425/
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
│
├── docs/                        # Documentation & Reports
│   ├── RelatorioCR.docx         # Final Report
│   └── TP2_IIA_2425.pdf         # Project Assignment Brief
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
│   ├── rede_top_73_162459.mat   # Accuracy: 73%
│   ├── rede_top_64_152746.mat   # Accuracy: 64%
│   └── rede_top_62_162036.mat   # Accuracy: 62%
│
├── draw/                        # Hand-drawn dataset for robustness testing
├── start/                       # Small initial dataset (5 per class)
├── train/                       # Main training dataset (50 per class)
└── test/                        # Testing dataset (10 per class)
(Note: Subdirectories for shapes—circle, kite, etc.—exist inside each dataset folder)🔧 PrerequisitesTo run this project, you need:MATLAB R2018b or higher.Deep Learning Toolbox (formerly Neural Network Toolbox).Image Processing Toolbox.📊 DatasetsThe model is trained and validated on four distinct datasets to ensure robustness:DatasetDescriptionCount (Images)PurposeStart5 images/class30Initial prototyping and quick testing.Train50 images/class300Main dataset for supervised learning.Test10 images/class60Unseen data for final evaluation.Draw5 images/class30Hand-drawn shapes to test generalization.Classes: Circle, Kite, Parallelogram, Square, Trapezoid, Triangle.🔍 Image Processing PipelineBefore feeding data into the network, all images undergo a standardized preprocessing pipeline:Ingestion: Load PNG images.Grayscale Conversion: RGB to Grayscale (if applicable).Resizing: Downsampling to 25x25 pixels.Binarization: Thresholding to convert into binary (black/white) images.Vectorization (Flattening): The 25x25 matrix is converted into a 625x1 column vector to match the input layer of the network.🚀 Usage Guide1. Data PreparationThe data loading functions (convertImage_*.m) automatically handle reading, resizing, and vectorization.Inputs: 625xN Matrix (N = number of images).Targets: 6xN Matrix (One-hot encoded vectors).2. Training & Evaluation StepsEnsure your MATLAB path includes the scripts/ folder.A. Initial Prototyping (Start Dataset)Explores basic topologies using a small dataset.Matlabcd scripts
train_start_a
B. Main Training (Train Dataset)Trains on the full dataset, iterating through:Topologies (neuron count/layers).Activation functions (tansig, logsig, purelin).Training algorithms (trainlm, traingd, etc.).Matlabtrain_train_b
C. Advanced EvaluationEvaluates trained models against the test set and combines datasets for maximum training data.Matlabtrain_test_c_i      % Evaluate on Test set
train_test_c_ii     % Retrain including Test data
train_all_c_iii     % Train on ALL available data (Start+Train+Test)
D. Real-world Test (Hand-Drawn)Tests the model's ability to recognize imperfect, hand-drawn shapes.Matlabtest_draw_d
3. GUI ApplicationFor an interactive experience, run the App Designer file:Matlabapp
🛠️ Network Architecture ExperimentsThe project analyzed several configurations to find the optimal balance between accuracy and computational cost.Topologies:Single Hidden Layer: [10], [20], [30] neurons.Deep Architectures: [20 20], [20 20 20], [30 20 10].Activation Functions:Hidden Layers: tansig (Hyperbolic Tangent), logsig (Log-Sigmoid).Output Layer: purelin (Linear), logsig.Training Algorithms:trainlm: Levenberg-Marquardt (Fastest, memory-intensive).traingd: Gradient Descent.traingda: Gradient Descent with Adaptive Learning Rate.

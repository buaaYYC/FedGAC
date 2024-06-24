# Introduction
FedCEA is designed to tackle the challenges of statistical heterogeneity in federated learning. It introduces a personalized FL method that reduces computational and communication overhead by focusing on Critical Learning Periods (CLP) for client participation. The Adaptive Initialization of Local Models (AILM) module enhances global model generalization, while dynamic training parameter adjustment ensures efficiency. Additionally, FedCEA employs a compression method to minimize communication costs. Extensive experiments demonstrate that FedCEA achieves superior accuracy and communication efficiency compared to state-of-the-art methods.
# Usage 
1. Requirement: Ubuntu 20.04, Python v3.5+, Pytorch and CUDA environment
2. "./fedcea_main.py" is about configurations and the basic Federated Learning framework
3. "./Sims.py" describes the simulators for clients and central server
4. "./Utils.py" contains all necessary functions and discusses how to get training and testing data
5. "./Settings.py" describes the necessary packages and settings
6. "./AILM.py" is the implementation of AILM algorithm
7. "./data_preprocessing.py" is the data preprocessing script for CIFAR-10\CIFAR-100\FashionMNIST\shakespeare dataset
8. Folder "./Models" includes codes for AlexNet, VGG-11, ResNet-18 and LSTM
9. Folder "./Optim" includes codes for FedProx, VRL-SGD, FedNova
10. Folder "./Comp_FIM" is the library to calculate Fisher Information Matrix (FIM)

# Train and evaluate FedCEA
 1. Use "./fedcea_main.py" to run results or ./run_fedcea.sh to run experiments
 2. The results will be saved in "./log" folder

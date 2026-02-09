# 3D-CNN-Transformer-Fusion-for-Spanish-Sign-Language-Recognition-


SignaMed-SSL: Temporal Self-Supervised Learning for Medical Sign Language
This repository contains a reproducible implementation of a self-supervised learning framework for medical sign language video representation learning. The approach is based on a compact 3D Convolutional Neural Network (3D CNN) trained with a binary temporal pretext task that distinguishes between temporally ordered and temporally shuffled sign language video clips.
The framework is evaluated on the SignaMed SWL-LSE dataset and is designed to learn spatiotemporal representations without relying on gloss-level annotations, making it suitable for low-resource and domain-specific settings such as healthcare communication.
 
Overview
The goal of this project is to learn meaningful spatiotemporal representations from sign language videos using a self-supervised objective. The model is trained to discriminate between naturally ordered sign sequences and temporally corrupted versions, encouraging the network to capture intrinsic motion dynamics.
The pipeline uses short RGB video clips and a lightweight 3D CNN backbone to balance representational power and computational efficiency.
 
Dataset
The experiments are conducted on the SignaMed SWL-LSE dataset, which contains reference videos of medically relevant Spanish Sign Language signs recorded under controlled conditions.
Dataset characteristics:
•	300 reference videos
•	187 health-related glosses
•	Domain-specific medical vocabulary
•	High visual and linguistic consistency
The dataset is downloaded automatically when the training script is executed.
Dataset portal:
https://signamed.web.app
 
Method
The proposed framework is based on a binary temporal self-supervised task:
•	Ordered clips preserve the original temporal sequence
•	Shuffled clips contain the same frames in a randomized temporal order
Each video contributes two training samples, doubling the dataset size without duplicating stored data. Temporal shuffling is applied dynamically during data loading.
Key design choices include:
•	Uniform sampling of 24 frames per video
•	Spatial-only pooling to preserve temporal resolution
•	Moderate spatial augmentation without altering temporal order
•	A compact 3D CNN with regularization and dropout
•	Fixed training schedule without early stopping
 
Repository Structure
.
├── train_temporal_ssl.py
├── README.md
├── signamed_temporal_best.keras
├── signamed_temporal_binary_model.keras
├── signamed_temporal_binary_metadata.npz
Model files and metadata are generated automatically after training.
 
Installation
The code requires Python 3.8 or later and the following libraries:
•	TensorFlow
•	NumPy
•	Pandas
•	OpenCV
•	Matplotlib
•	scikit-learn
Install dependencies using:
pip install tensorflow numpy pandas opencv-python matplotlib scikit-learn
 
Running the Code
To run the full self-supervised training pipeline:
python train_temporal_ssl.py
The script performs the following steps:
1.	Downloads and verifies dataset files
2.	Builds training, validation, and test splits
3.	Constructs the binary temporal dataset
4.	Trains the 3D CNN for 30 epochs
5.	Evaluates performance on a held-out test set
6.	Saves the trained model and metadata
 
Results
The trained model demonstrates stable convergence and strong generalization on the temporal discrimination task.
•	Peak validation accuracy: 95.00%
•	Test accuracy: 86.67%
•	Task: Binary temporal order prediction
•	Inference throughput: 80+ clips per second on a single GPU
Balanced error rates indicate that the model learns temporal structure rather than exploiting superficial visual cues.
 
Intended Use
This repository focuses on self-supervised representation learning and is intended to serve as:
•	A reproducible baseline for temporal self-supervised learning in sign language
•	A foundation for downstream tasks such as medical sign classification
•	A reference implementation for annotation-efficient video learning
The code does not perform multi-class gloss recognition or sign language translation.
 
Notes and Limitations
•	The framework operates on RGB video only and does not use pose or skeleton features
•	Performance is evaluated on the self-supervised pretext task, not on downstream recognition
•	The dataset size is modest, reflecting realistic constraints in medical domains
 
License and Usage
This code is provided for research and educational purposes. Please follow the dataset license terms when using SignaMed SWL-LSE data.




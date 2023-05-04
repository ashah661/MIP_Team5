# MIP_Team5
Codebase of the work accomplished by Team 5 for ECE6780 Medical Image Processing - Spring 2023

This project aims to detect the Diabetic Retinopathy Severity Grade using the Retinal Fundus Color images. The project consists of 4 primary python scripts.
1. train-test-splitter.ipynb: This splits the dataset into train, validation and test sets. The ratio of each class is maintained in all the sets.
2. SP_EyePACS_v2.ipynb: This is the code for training the EyePACS dataset from scratch using weighted class entropy. The model is initialized with ImageNet weights.
3. AS_IDRiD_v2.ipynb: This is the code for training the IDRiD dataset from scratch using weighted class entropy. The model is initialized with ImageNet weights.
4. EyePACS-IDRiD_Finetune.ipynb: This file contains the code to finetune the model trained on EyePACS dataset. The last best model from the training on EyePACS dataset is loaded and used to train on IDRiD dataset for a fewer number of epochs.
5. AS_v3 is a script to generate performance of all the models on the test dataset to generate predictability plots
STEPS TO RUN
After uploading the dataset, first, run the train-test-splitter to split the dataset into the train, validation, and test sets. You might have to change the paths to your dataset folder. The result of running this script is the generation of 3 csv files (train, val, test) which contains the image names and labels corresponding to them.
The AS_IDRiD_v2 (for IDRiD dataset) or SP_EyePACS_v2 script (for EyePACS dataset) can be then run to create and train the model for the respective dataset. Both these files consist of the same key components:
1.	Visualizing and calculating the class imbalance
2.	Defining a function to pre-process the image using CLAHE algorithm
3.	Defining a custom dataset class to create our training, validation, and testing dataset
4.	Initializing train, val, and test dataloaders. In all the dataloaders, the images are normalized and resized to 512x512. In the train dataloader, image processing and augmentation techniques are implemented to improve the training process. 
5.	Model definition and model weights initialization 
6.	Trainer loop (training and validation loss calculation and writing it in a .txt file)
7.	Saving checkpoints for the best and last model based on lowest validation loss
8.	Testing the model on the test dataloader to calculate its performance using the metrics defined – F1 score, Cohen’s Kappa, and Accuracy
9.	Generating the plot for training and validation loss over the number of epochs the model was trained. 

Files generated after running the script:
  •	training_loss.txt
  •	validation_loss.txt
  •	(dataset)_best_(technique).pt
  •	(dataset)_last_(technique).pt
  •	(technique)_trainLossValLoss.png

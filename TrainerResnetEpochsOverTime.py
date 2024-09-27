import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch import nn
from torchvision import models
import torch
import ModifiedDataLoader
import time

# 7 models for each bin of 50
# import matplotlib.pyplot as plt
# use
"""
scp -F /dev/null -o "ProxyCommand sft proxycommand %h" -oStrictHostKeyChecking=no PycharmProjects\pythonProject1\TrainerResnetEpochsOverTime.py beehive:
"""

import numpy as np


def cross_entropy_loss(y_true, y_pred):
    """
    Calculate the cross-entropy loss between two arrays.

    Parameters:
    y_true (numpy array): Array of true labels (one-hot encoded).
    y_pred (numpy array): Array of predicted probabilities.

    Returns:
    float: Cross-entropy loss.

    Used only when reporting results; not during training
    torch's inbuilt cross entropy loss is used while training
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Calculate cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss


# DataModule class
class TrainingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=2, first_frame = 250, numFrames = 250):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # move to dataloader
            transforms.ToTensor(),  # Converts to float and normalizes between 0 and 1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #proper normalization values
            """more information about the values above found here: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2 """
        ])
        self.num_frames = numFrames
        self.first_frame = first_frame

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ModifiedDataLoader. LiveCellImageDataset(
                wells=['B3', 'B4', 'B5', 'B7', 'B8', 'B9', 'E3', 'E4', 'E5'], startFrame = self.first_frame, numFrames = self.num_frames)
            # Apply transform to each image in the dataset
            self.train_dataset.transform = self.transform
        if stage == 'predict':
            self.predict_dataset = ModifiedDataLoader.LiveCellImageDataset(
                wells=['B3', 'B4', 'B5', 'B7', 'B8', 'B9', 'E3', 'E4', 'E5'], startFrame = self.first_frame, numFrames = self.num_frames)
            # Apply transform to each image in the dataset
            self.predict_dataset.transform = self.transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=2, shuffle=False)


class TestingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=2, first_frame = 250, numFrames = 350):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts to float and normalizes between 0 and 1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.first_frame = first_frame
        self.num_frames = numFrames

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ModifiedDataLoader.LiveCellImageDataset(wells=['B6', 'B10', 'E6'], startFrame = self.first_frame, numFrames = self.num_frames)
            # Apply transform to each image in the dataset
            self.train_dataset.transform = self.transform
        if stage == 'predict':
            self.predict_dataset = ModifiedDataLoader.LiveCellImageDataset(wells=['B6', 'B10', 'E6'], startFrame = self.first_frame, numFrames = self.num_frames)
            # Apply transform to each image in the dataset
            self.predict_dataset.transform = self.transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=2, shuffle=False)


# Model class
class CellModel(pl.LightningModule):
    def __init__(self, num_classes=3):
        super(CellModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1) #change model architechture here
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # replacing final layer with my own output

    def forward(self, x):
        x = x.float()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels, _ = batch
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, labels)
        #print('batch index: ' + str(batch_idx))
        #print('loss: ' + str(loss))
        #self.log('train_loss', loss)  # log metrics and google how to access
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def predict_step(self, batch, batch_idx):
        images, _, filenames = batch
        outputs = self(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities, filenames


# Main script
if __name__ == '__main__':
    start = time.time()
    modelName = 'thisEpochNew2.pth'

    accuracy = []
    accuracyTraining = []
    crossEntropyTrain = []
    crossEntropyTest = []
    numFrames = 350
    firstFrame = 0

    # Initialize the DataModule
    data_module = TrainingDataModule(first_frame=firstFrame, numFrames=numFrames)
    test_module = TestingDataModule(first_frame=firstFrame, numFrames=numFrames)

    # Initialize the model
    model = CellModel(num_classes=3)
    
    # Initialize the Trainer


    for a in range(10):
        # Train the model

        trainer = Trainer(max_epochs=3, devices=1, accelerator='cuda')
        trainer.fit(model, datamodule=data_module)

        torch.save(model.state_dict(), modelName)
        print("Model trained and saved as " + modelName)

        # Set up the DataModules for prediction
        data_module.setup(stage='predict')
        test_module.setup(stage='predict')

        # Run inference and print predictions
        probabilitiesTraining = []
        print("Training data predictions:")
        predictions = trainer.predict(model, datamodule=data_module)
        print('num preds: ' + str(len(predictions)))
        for batch in predictions:
            probs, filenames = batch
            for i in range(len(filenames)):
                print('Frame number: ' + str(i % numFrames))
                probabilitiesTraining.append(probs[i])
                print(f'{filenames[i]}: {probs[i].tolist()}')

        probabilities = []
        print("Testing data predictions:")
        predictions = trainer.predict(model, datamodule=test_module)
        for batch in predictions:
            probs, filenames = batch
            for i in range(len(filenames)):
                print('Frame number: ' + str(i % numFrames))
                probabilities.append(probs[i])
                print(f'{filenames[i]}: {probs[i].tolist()}')

        SH = [1, 0, 0]
        CUL5 = [0, 1, 0]
        RAS2 = [0, 0, 1]

        SH_copies = [SH] * numFrames
        CUL5_copies = [CUL5] * numFrames
        RAS2_copies = [RAS2] * numFrames

        # Concatenate the lists into a single list
        combined_list = SH_copies + CUL5_copies + RAS2_copies
        combined_list_training = SH_copies*3 + CUL5_copies*3 + RAS2_copies*3

        # Convert the list into a numpy array
        y_true = np.array(combined_list)
        y_true_training = np.array(combined_list_training)

        y_pred = np.array(probabilities)
        y_pred_training = np.array(probabilitiesTraining)

        confusionMatrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        confusionMatrixTraining = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for k in range(3):
            for i in range(numFrames):
                for j in range(3):
                    if probabilities[k * numFrames + i][j] >= 0.5:
                        confusionMatrix[k][j] += 1

        for k in range(9):
            for i in range(numFrames):
                for j in range(3):
                    if probabilitiesTraining[k * numFrames + i][j] >= 0.5:
                        confusionMatrixTraining[k//3][j] += 1

        # y_pred = np.array([[0.9, 0.1, 0], [0.1, 0.9, 0.0], [0, 0.1, 0.9]])

        loss = cross_entropy_loss(y_true, y_pred)
        lossTraining = cross_entropy_loss(y_true_training, y_pred_training)
        crossEntropyTrain.append(lossTraining)
        crossEntropyTest.append(loss)

        print(f"Cross-Entropy Loss: {loss}")

        print('Confusion Matrix: ')
        print(confusionMatrix)

        val = 0
        val += confusionMatrix[0][0]
        val += confusionMatrix[1][1]
        val += confusionMatrix[2][2]
        val /= numFrames*3
        print(val)
        accuracy.append(val)

        val = 0
        val += confusionMatrixTraining[0][0]
        val += confusionMatrixTraining[1][1]
        val += confusionMatrixTraining[2][2]
        val /= numFrames*9
        print(val)
        accuracyTraining.append(val)

        print('Training Accuracies: ')
        print(accuracyTraining)
        print('Testing Accuracies: ')
        print(accuracy)
        print('Training Cross Entropy Loss: ')
        print(crossEntropyTrain)
        print('Testing Cross Entropy Loss: ')
        print(crossEntropyTest)

        model.load_state_dict(torch.load(modelName))

    print('accuracy testing over time (each val corresponds to 10 epochs)')
    print(accuracy)
    print('accuracy training: ')
    print(accuracyTraining)
    print('Training Cross Entropy Loss: ')
    print(crossEntropyTrain)
    print('Testing Cross Entropy Loss: ')
    print(crossEntropyTest)
    end = time.time()
    print('time taken to train/run: ')
    print(end-start)
    print("Model trained and saved as " + modelName)

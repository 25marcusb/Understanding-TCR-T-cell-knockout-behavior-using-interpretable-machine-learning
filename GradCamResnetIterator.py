import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchvision import models
import ModifiedDataLoader

"""Generates GradCam visualizations"""

# Define the custom model
class CellModel(pl.LightningModule):
    def __init__(self, num_classes=3):
        super(CellModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # replacing final layer with my own output

        # Placeholder for the gradients
        self.gradients = None

    def forward(self, x):
        x = x.float()
        return self.model(x)

    # Hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


# DataModule class
class TrainingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=2, startFrame=250):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.startFrame = startFrame

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ModifiedDataLoader.LiveCellImageDataset(
                wells=['B6', 'B10', 'E6'], startFrame=self.startFrame)
            self.train_dataset.transform = self.transform
        if stage == 'predict':
            self.predict_dataset = ModifiedDataLoader.LiveCellImageDataset(
                wells=['B6', 'B10', 'E6'], startFrame=self.startFrame)
            self.predict_dataset.transform = self.transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1, shuffle=False)


class TestingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=2, first_frame=250):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts to float and normalizes between 0 and 1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.first_frame = first_frame

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ModifiedDataLoader.LiveCellImageDataset(wells=['B6', 'B10', 'E6'],
                                                                               startFrame=self.first_frame)
            # Apply transform to each image in the dataset
            self.train_dataset.transform = self.transform
        if stage == 'predict':
            self.predict_dataset = ModifiedDataLoader.LiveCellImageDataset(wells=['B6', 'B10', 'E6'],
                                                                                 startFrame=self.first_frame)
            # Apply transform to each image in the dataset
            self.predict_dataset.transform = self.transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=2, shuffle=False)


# Initialize the DataModule and model
print('test')
data_module = TrainingDataModule(startFrame=100)
data_module.setup(stage='predict')
model = CellModel(num_classes=3)
model.load_state_dict(torch.load("model100resnetTest.pth"))
model.eval()
dataloader = data_module.predict_dataloader()

for j, (img, _, _) in enumerate(dataloader):
    # Ensure the image tensor is of type Float
    img = img.float()


    # Register hook to capture gradients at layer4[2].conv3
    def forward_hook(module, input, output):
        model.activations = output
        output.register_hook(model.activations_hook)


    hook_handle = model.model.layer4[2].register_forward_hook(forward_hook)

    framesPerWell = 50

    # Forward pass to get the prediction
    pred = model(img)

    if j < framesPerWell:
        category = 'SH'
        classNum = 0
    elif j < framesPerWell * 2:
        category = 'CU'
        classNum = 1
    elif j < framesPerWell * 3:
        category = 'RA'
        classNum = 2
    else:
        raise ValueError('j larger than expected')

    # Get the gradient of the output with respect to the parameters of the model
    pred[:, classNum].backward()  # Assuming class 0 for demonstration, modify as necessary

    # Pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # Ensure gradients are not None
    if gradients is None:
        raise ValueError("Gradients are not captured properly. Ensure the hook is correctly registered.")

    # Pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Get the activations of the last convolutional layer
    activations = model.activations.detach()

    # Weight the channels by corresponding gradients
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # ReLU on top of the heatmap
    heatmap = np.maximum(heatmap, 0)

    # Normalize the heatmap
    heatmap /= torch.max(heatmap)

    # Draw the heatmap
    plt.matshow(heatmap.squeeze())
    plt.savefig('heatmap.png')

    # Convert the tensor image to a format suitable for OpenCV
    img = img.squeeze().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    img = np.uint8(255 * img)

    heatmap = cv2.resize(np.array(heatmap), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    filename = './' + category + str(j % 50) + 'map.jpg'

    cv2.imwrite(filename, superimposed_img)
    print(filename)

    # Clean up hooks
    hook_handle.remove()

print('finished without errors')

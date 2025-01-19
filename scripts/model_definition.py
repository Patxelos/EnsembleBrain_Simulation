import torch
import torch.nn as nn
from torchvision import models

# Define the baseline CNN model
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        # Load the pretrained VGG16 model
        vgg16 = models.vgg16(pretrained=True)
        
        # Extract convolutional layers (features)
        self.feature_extractor = vgg16.features
        
        # Add custom fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the feature maps
            nn.Linear(512 * 7 * 7, 4096),  # Fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),  # Fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # Final output layer
        )
        
    def forward(self, x):
        # Pass the input through the feature extractor (backbone)
        features = self.feature_extractor(x)
        # Pass the extracted features through the classifier
        output = self.classifier(features)
        return output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# a neural network class that uses a pre-trained ResNet-50 model as a feature extractor 
class FeatureExtractor(nn.Module):
    # constructor that initializes the model 
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Load the pre-trained ResNet-50 model with weights trained on the ImageNet dataset
        basemodel = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = basemodel.fc.in_features
        basemodel.fc = nn.Identity()
        self.basemodel = basemodel
        self.embeddings_layer = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.Dropout(0.2)
        )
         # Freeze all ResNet backbone layers
        for param in self.basemodel.parameters():
            param.requires_grad = False

        # Keep projection/embedding layer trainable
        for param in self.embeddings_layer.parameters():
            param.requires_grad = True
       
    def forward(self, x):
        x = self.basemodel(x)
        x=self.embeddings_layer(x)
        x = F.normalize(x, p=2, dim=1)
        return x

        
         
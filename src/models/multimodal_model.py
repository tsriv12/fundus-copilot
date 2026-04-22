import torch
import torch.nn as nn
from torchvision import models


class FundusMultimodalModel(nn.Module):
    def __init__(self, metadata_dim=10, metadata_hidden_dim=32, fusion_hidden_dim=128):
        super().__init__()

        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        image_feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.image_encoder = backbone

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(metadata_hidden_dim, metadata_hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + metadata_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, image, metadata):
        image_features = self.image_encoder(image)
        metadata_features = self.metadata_encoder(metadata)
        fused = torch.cat([image_features, metadata_features], dim=1)
        logits = self.classifier(fused)
        return logits

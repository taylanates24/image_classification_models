import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Calculate Cross Entropy Loss
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probabilities for the targets
        pt = torch.exp(-CE_loss)

        # Calculate Focal Loss
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * CE_loss

        return focal_loss.mean()
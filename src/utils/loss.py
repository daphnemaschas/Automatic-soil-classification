"""
SpaceNet 8 Dice Loss Utility Module.

This module provides an implementation of the dice loss used as a complement of the cross entropy loss to train the model.
"""

import torch

def dice_loss(pred, target, num_classes):
    target_oh = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    prob = torch.softmax(pred, dim=1)
    
    intersection = (prob * target_oh).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()
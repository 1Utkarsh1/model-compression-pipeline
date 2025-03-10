#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline model module for loading and training pretrained models.
"""

import os
import time
import logging
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VisionTransformer(nn.Module):
    """
    Simple wrapper for Vision Transformer models.
    """
    def __init__(self, model_name: str, num_classes: int):
        """
        Initialize a Vision Transformer model.
        
        Args:
            model_name (str): Name of the ViT model variant
            num_classes (int): Number of output classes
        """
        super(VisionTransformer, self).__init__()
        
        # Available model variants
        vit_variants = {
            'vit_b_16': models.vit_b_16,
            'vit_b_32': models.vit_b_32,
            'vit_l_16': models.vit_l_16,
            'vit_l_32': models.vit_l_32,
            'vit_h_14': models.vit_h_14,
        }
        
        if model_name not in vit_variants:
            raise ValueError(f"Unsupported ViT variant: {model_name}. "
                           f"Available variants: {', '.join(vit_variants.keys())}")
        
        # Load the pretrained model
        model_fn = vit_variants[model_name]
        
        # Get appropriate weights
        weights = None
        if hasattr(models, f"{model_name.upper()}_Weights"):
            weights_enum = getattr(models, f"{model_name.upper()}_Weights")
            weights = weights_enum.IMAGENET1K_V1
        
        # Load model
        self.model = model_fn(weights=weights)
        
        # Replace the head with a new one for the target number of classes
        if hasattr(self.model, 'heads'):
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Could not find classification head in {model_name}")
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)


def load_baseline_model(model_name: str, num_classes: int) -> Tuple[nn.Module, optim.Optimizer, nn.Module]:
    """
    Load a pretrained model and adapt it for the given number of classes.
    
    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of classes for the classification task
        
    Returns:
        tuple: (model, optimizer, criterion)
    """
    logger.info(f"Loading {model_name} model with {num_classes} output classes...")
    
    # Initialize model based on the specified architecture
    if model_name.lower() == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name.lower() == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name.lower() == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name.lower() == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name.lower() == 'mobilenet':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_name.lower().startswith('efficientnet'):
        # For EfficientNet, we need to handle different variants
        if model_name.lower() == 'efficientnet-b0' or model_name.lower() == 'efficientnet':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name.lower() == 'efficientnet-b1':
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        elif model_name.lower() == 'efficientnet-b2':
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        elif model_name.lower() == 'efficientnet-b3':
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
        
        # Replace the classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_name.lower().startswith('vit'):
        # For Vision Transformer models
        model = VisionTransformer(model_name, num_classes)
    
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Move model to available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    logger.info(f"Model loaded successfully on {device}")
    
    return model, optimizer, criterion


def train_baseline_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    learning_rate: float,
    save_path: str
) -> Dict[str, Any]:
    """
    Train the baseline model.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (optim.Optimizer): Optimizer to use
        criterion (nn.Module): Loss function
        epochs (int): Number of training epochs
        learning_rate (float): Initial learning rate
        save_path (str): Path to save the trained model
        
    Returns:
        dict: Training history
    """
    logger.info(f"Training baseline model for {epochs} epochs...")
    
    device = next(model.parameters()).device
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=10, verbose=True
    )
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': train_loss / train_total,
                'acc': 100. * train_correct / train_total
            })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': val_loss / val_total,
                    'acc': 100. * val_correct / val_total
                })
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            logger.info(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                   f"Time: {epoch_time:.2f}s")
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load the best model
    model.load_state_dict(torch.load(save_path))
    
    return history


def test_baseline_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """
    Test the baseline model.
    
    Args:
        model (nn.Module): Model to test
        test_loader (DataLoader): Test data loader
        
    Returns:
        dict: Test metrics
    """
    logger.info("Testing baseline model...")
    
    device = next(model.parameters()).device
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for inputs, targets in test_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            test_pbar.set_postfix({
                'loss': test_loss / test_total,
                'acc': 100. * test_correct / test_total
            })
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100. * test_correct / test_total
    
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }


if __name__ == '__main__':
    # Test the baseline model
    import sys
    import os
    
    # Add the project root directory to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from src.data.data_loader import load_dataset
    
    logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    train_loader, val_loader, test_loader = load_dataset('cifar10', batch_size=64)
    
    # Load model
    model, optimizer, criterion = load_baseline_model('resnet18', 10)
    
    # Train for just a few epochs as a test
    save_path = 'test_baseline.pth'
    history = train_baseline_model(
        model, train_loader, val_loader, optimizer, criterion,
        epochs=2, learning_rate=0.01, save_path=save_path
    )
    
    # Test the model
    test_metrics = test_baseline_model(model, test_loader)
    print(f"Test metrics: {test_metrics}")
    
    # Clean up test model file
    if os.path.exists(save_path):
        os.remove(save_path) 
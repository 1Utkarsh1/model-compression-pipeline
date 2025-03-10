#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model pruning module for applying weight pruning techniques.
"""

import os
import logging
import copy
from typing import Dict, List, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.baseline import load_baseline_model, train_baseline_model

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count the total and non-zero parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, non_zero_params)
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_zero_params = sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)
    
    return total_params, non_zero_params


def get_prunable_layers(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Get a dictionary of prunable layers (convolutional and linear) in the model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        dict: Dictionary of prunable layers {name: module}
    """
    prunable_layers = {}
    
    for name, module in model.named_modules():
        # We will prune weights in Conv2d and Linear layers
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable_layers[name] = module
    
    logger.info(f"Found {len(prunable_layers)} prunable layers")
    return prunable_layers


def apply_magnitude_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    Apply global magnitude pruning to the model.
    
    Args:
        model (nn.Module): PyTorch model
        amount (float): Fraction of weights to prune (0.0 to 1.0)
        
    Returns:
        nn.Module: Pruned model
    """
    logger.info(f"Applying magnitude pruning with pruning rate {amount:.2f}")
    
    # Get parameters to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    # Check pruning results
    total_params, non_zero_params = count_parameters(model)
    sparsity = 1.0 - (non_zero_params / total_params)
    
    logger.info(f"Pruning completed. Model sparsity: {sparsity:.4f}")
    logger.info(f"Parameters: {non_zero_params:,}/{total_params:,}")
    
    return model


def apply_random_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    Apply random pruning to the model.
    
    Args:
        model (nn.Module): PyTorch model
        amount (float): Fraction of weights to prune (0.0 to 1.0)
        
    Returns:
        nn.Module: Pruned model
    """
    logger.info(f"Applying random pruning with pruning rate {amount:.2f}")
    
    # Get parameters to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global unstructured random pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=amount
    )
    
    # Check pruning results
    total_params, non_zero_params = count_parameters(model)
    sparsity = 1.0 - (non_zero_params / total_params)
    
    logger.info(f"Pruning completed. Model sparsity: {sparsity:.4f}")
    logger.info(f"Parameters: {non_zero_params:,}/{total_params:,}")
    
    return model


def apply_structured_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    Apply structured pruning to convolutional layers in the model.
    
    Args:
        model (nn.Module): PyTorch model
        amount (float): Fraction of channels/filters to prune (0.0 to 1.0)
        
    Returns:
        nn.Module: Pruned model
    """
    logger.info(f"Applying structured pruning with pruning rate {amount:.2f}")
    
    # Apply structured pruning to each convolutional layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    
    # Check pruning results
    total_params, non_zero_params = count_parameters(model)
    sparsity = 1.0 - (non_zero_params / total_params)
    
    logger.info(f"Pruning completed. Model sparsity: {sparsity:.4f}")
    logger.info(f"Parameters: {non_zero_params:,}/{total_params:,}")
    
    return model


def fine_tune_pruned_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001
) -> nn.Module:
    """
    Fine-tune the pruned model to recover accuracy.
    
    Args:
        model (nn.Module): Pruned model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): Number of fine-tuning epochs
        learning_rate (float): Learning rate for fine-tuning
        
    Returns:
        nn.Module: Fine-tuned model
    """
    logger.info(f"Fine-tuning pruned model for {epochs} epochs with learning rate {learning_rate}")
    
    # Set up optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Get temporary path to save the best model during fine-tuning
    temp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            'results', 'temp_pruned_model.pth')
    
    # Fine-tune the model
    history = train_baseline_model(
        model, train_loader, val_loader, optimizer, criterion,
        epochs=epochs, learning_rate=learning_rate, save_path=temp_path
    )
    
    # Load the best model
    model.load_state_dict(torch.load(temp_path))
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return model


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Make pruning permanent by replacing pruned parameters with actual zeros.
    
    Args:
        model (nn.Module): Pruned model
        
    Returns:
        nn.Module: Model with permanent pruning
    """
    logger.info("Making pruning permanent")
    
    # Create a deep copy of the model
    permanent_model = copy.deepcopy(model)
    
    # Make pruning permanent by removing the reparameterization
    for name, module in permanent_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
    
    return permanent_model


def prune_model(
    model_path: str,
    prune_rate: float,
    method: str = 'magnitude',
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    fine_tune_epochs: int = 10
) -> nn.Module:
    """
    Main function to prune a model.
    
    Args:
        model_path (str): Path to the model checkpoint
        prune_rate (float): Fraction of weights to prune (0.0 to 1.0)
        method (str): Pruning method ('magnitude', 'random', 'structured')
        train_loader (DataLoader): Training data loader for fine-tuning
        val_loader (DataLoader): Validation data loader for fine-tuning
        fine_tune_epochs (int): Number of fine-tuning epochs
        
    Returns:
        nn.Module: Pruned and fine-tuned model
    """
    # Load the model
    logger.info(f"Loading model from {model_path}")
    
    # We need to determine the number of classes from the model
    # For simplicity, we'll load a temporary model based on ResNet50 with 1000 classes
    # and then adjust based on the actual model's output layer
    temp_model, _, _ = load_baseline_model('resnet50', num_classes=1000)
    
    # Load the state dict
    state_dict = torch.load(model_path)
    
    # Determine the number of classes from the output layer weights
    if 'fc.weight' in state_dict:  # ResNet architecture
        num_classes = state_dict['fc.weight'].size(0)
        model_name = 'resnet50'  # Assume ResNet50 by default
        
        # Check if it's another ResNet variant based on layer dimensions
        fc_in_features = state_dict['fc.weight'].size(1)
        if fc_in_features == 512:
            model_name = 'resnet18'
        elif fc_in_features == 1024:
            model_name = 'resnet34'
        elif fc_in_features == 2048:
            model_name = 'resnet50' or 'resnet101'  # Can't distinguish easily, assume ResNet50
            
    elif 'classifier.1.weight' in state_dict:  # MobileNet or EfficientNet
        num_classes = state_dict['classifier.1.weight'].size(0)
        # Assume MobileNet by default, but check for EfficientNet characteristics
        if any('blocks' in key for key in state_dict.keys()):
            model_name = 'efficientnet'
        else:
            model_name = 'mobilenet'
    else:
        raise ValueError("Unable to determine model architecture from state dict")
    
    # Load the actual model with the correct number of classes
    model, _, _ = load_baseline_model(model_name, num_classes)
    model.load_state_dict(state_dict)
    
    # Count parameters before pruning
    total_before, non_zero_before = count_parameters(model)
    logger.info(f"Model before pruning: {non_zero_before:,}/{total_before:,} parameters")
    
    # Apply the appropriate pruning method
    if method.lower() == 'magnitude':
        pruned_model = apply_magnitude_pruning(model, prune_rate)
    elif method.lower() == 'random':
        pruned_model = apply_random_pruning(model, prune_rate)
    elif method.lower() == 'structured':
        pruned_model = apply_structured_pruning(model, prune_rate)
    else:
        raise ValueError(f"Unsupported pruning method: {method}")
    
    # Fine-tune the pruned model if data loaders are provided
    if train_loader is not None and val_loader is not None:
        pruned_model = fine_tune_pruned_model(
            pruned_model, train_loader, val_loader, fine_tune_epochs
        )
    
    # Make pruning permanent
    pruned_model = make_pruning_permanent(pruned_model)
    
    # Count parameters after pruning
    total_after, non_zero_after = count_parameters(pruned_model)
    sparsity = 1.0 - (non_zero_after / total_after)
    
    logger.info(f"Final pruned model: {non_zero_after:,}/{total_after:,} parameters")
    logger.info(f"Achieved sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    logger.info(f"Compression ratio: {total_after/non_zero_after:.2f}x")
    
    return pruned_model


if __name__ == '__main__':
    # Test the pruning module
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Load a small model for testing
    model, _, _ = load_baseline_model('resnet18', 10)
    
    # Save the model temporarily
    temp_path = 'temp_model.pth'
    torch.save(model.state_dict(), temp_path)
    
    # Count parameters before pruning
    total_before, non_zero_before = count_parameters(model)
    print(f"Model before pruning: {non_zero_before:,}/{total_before:,} parameters")
    
    # Test pruning with a small rate
    pruned_model = prune_model(temp_path, prune_rate=0.5, method='magnitude')
    
    # Count parameters after pruning
    total_after, non_zero_after = count_parameters(pruned_model)
    print(f"Model after pruning: {non_zero_after:,}/{total_after:,} parameters")
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path) 
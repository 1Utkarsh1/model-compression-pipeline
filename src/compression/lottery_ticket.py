#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of the Lottery Ticket Hypothesis for model compression.

The Lottery Ticket Hypothesis (Frankle & Carbin, 2019) suggests that dense, randomly 
initialized neural networks contain subnetworks (winning tickets) that - when trained 
in isolation - reach test accuracy comparable to the original network in a similar 
number of iterations.

Reference: https://arxiv.org/abs/1803.03635
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
from src.compression.pruning import count_parameters, get_prunable_layers, make_pruning_permanent

logger = logging.getLogger(__name__)


def save_initial_weights(model: nn.Module, save_path: str) -> None:
    """
    Save the initial weights of a model before training.
    
    Args:
        model (nn.Module): PyTorch model with initial weights
        save_path (str): Path to save the initial weights
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), save_path)
    logger.info(f"Initial weights saved to {save_path}")


def reset_pruned_model(model: nn.Module, initial_weights_path: str) -> nn.Module:
    """
    Reset the weights of a pruned model to their initial values, keeping the pruning mask.
    
    Args:
        model (nn.Module): Pruned PyTorch model
        initial_weights_path (str): Path to the initial weights
        
    Returns:
        nn.Module: Model with initial weights but pruning mask preserved
    """
    # Load initial weights
    initial_state_dict = torch.load(initial_weights_path)
    
    # Create a dictionary to store pruning masks
    masks = {}
    
    # Extract masks from the pruned model
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                masks[name] = module.weight_mask.clone()
    
    # Load initial weights
    model.load_state_dict(initial_state_dict)
    
    # Reapply masks to the model
    for name, module in model.named_modules():
        if name in masks:
            # Create a temporary parameter
            prune.custom_from_mask(module, name='weight', mask=masks[name])
    
    logger.info("Model reset to initial weights with pruning mask preserved")
    return model


def find_lottery_ticket(
    model_path: str,
    initial_weights_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prune_percent: float = 0.2,
    iterations: int = 5,
    epochs_per_iteration: int = 10,
    learning_rate: float = 0.01
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """
    Implement the Lottery Ticket Hypothesis to find a winning ticket.
    
    Args:
        model_path (str): Path to the pretrained model
        initial_weights_path (str): Path to the initial weights
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        prune_percent (float): Percentage of weights to prune in each iteration (0.0 to 1.0)
        iterations (int): Number of pruning iterations
        epochs_per_iteration (int): Training epochs per pruning iteration
        learning_rate (float): Learning rate for training
        
    Returns:
        tuple: (winning_ticket_model, iteration_metrics)
    """
    logger.info(f"Finding lottery ticket with {iterations} iterations, pruning {prune_percent*100}% each time")
    
    # Load the pretrained model
    state_dict = torch.load(model_path)
    
    # Determine model architecture and number of classes
    if 'fc.weight' in state_dict:  # ResNet
        num_classes = state_dict['fc.weight'].size(0)
        model_name = 'resnet50'  # Assume ResNet50 by default
    elif 'classifier.1.weight' in state_dict:  # MobileNet or EfficientNet
        num_classes = state_dict['classifier.1.weight'].size(0)
        model_name = 'mobilenet' if not any('blocks' in k for k in state_dict.keys()) else 'efficientnet'
    else:
        raise ValueError("Cannot determine model architecture from state dict")
    
    # Load model
    model, optimizer, criterion = load_baseline_model(model_name, num_classes)
    model.load_state_dict(state_dict)
    
    # Initialize tracking
    current_model = copy.deepcopy(model)
    iteration_metrics = []
    current_prune_rate = 0.0
    cumulative_prune_rate = 0.0
    
    # Iteratively prune and retrain
    for iteration in range(iterations):
        logger.info(f"Iteration {iteration+1}/{iterations}")
        
        # Calculate parameters before pruning
        total_params, non_zero_params = count_parameters(current_model)
        logger.info(f"Before pruning: {non_zero_params:,}/{total_params:,} parameters")
        
        # Apply pruning (except for the first iteration which uses the pretrained model)
        if iteration > 0:
            # Get prunable layers
            prunable_layers = get_prunable_layers(current_model)
            
            # Calculate pruning rate for this iteration (increase iteratively)
            current_prune_rate = prune_percent / (1 - cumulative_prune_rate)
            cumulative_prune_rate += prune_percent * (1 - cumulative_prune_rate)
            
            logger.info(f"Pruning {current_prune_rate*100:.2f}% of remaining weights")
            
            # Apply magnitude pruning to each layer
            parameters_to_prune = []
            for name, module in prunable_layers.items():
                parameters_to_prune.append((module, 'weight'))
            
            # Apply global unstructured pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=current_prune_rate
            )
            
            # Reset weights to initial values
            current_model = reset_pruned_model(current_model, initial_weights_path)
        
        # Count parameters after pruning
        total_params, non_zero_params = count_parameters(current_model)
        actual_sparsity = 1.0 - (non_zero_params / total_params)
        logger.info(f"After pruning: {non_zero_params:,}/{total_params:,} parameters")
        logger.info(f"Sparsity: {actual_sparsity*100:.2f}%")
        
        # Save temp path for the best model in this iteration
        results_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        temp_path = os.path.join(results_dir, 'results', f'lottery_ticket_iter{iteration+1}.pth')
        
        # Train the pruned model
        temp_optimizer = torch.optim.SGD(
            current_model.parameters(), 
            lr=learning_rate, 
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        history = train_baseline_model(
            current_model, 
            train_loader, 
            val_loader, 
            temp_optimizer, 
            criterion, 
            epochs=epochs_per_iteration, 
            learning_rate=learning_rate, 
            save_path=temp_path
        )
        
        # Evaluate on validation set
        current_model.load_state_dict(torch.load(temp_path))
        
        # Save iteration metrics
        iteration_metrics.append({
            'iteration': iteration + 1,
            'sparsity': actual_sparsity,
            'param_count': non_zero_params,
            'best_val_acc': max(history['val_acc']),
        })
        
        # Make pruning permanent for the next iteration
        if iteration < iterations - 1:
            current_model = make_pruning_permanent(current_model)
    
    # Return the winning ticket (best performing model with highest sparsity)
    logger.info("Lottery ticket search completed")
    return current_model, iteration_metrics


def apply_lottery_ticket_pruning(
    model_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prune_percent: float = 0.2,
    iterations: int = 5,
    epochs_per_iteration: int = 10,
    learning_rate: float = 0.01
) -> nn.Module:
    """
    Apply Lottery Ticket Hypothesis pruning to a model.
    
    Args:
        model_path (str): Path to the pretrained model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        prune_percent (float): Percentage of weights to prune in each iteration (0.0 to 1.0)
        iterations (int): Number of pruning iterations
        epochs_per_iteration (int): Training epochs per pruning iteration
        learning_rate (float): Learning rate for training
        
    Returns:
        nn.Module: Pruned model (winning ticket)
    """
    # Generate a path for saving initial weights
    results_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    initial_weights_path = os.path.join(results_dir, 'results', 'initial_weights.pth')
    
    # Load the model to initialize with random weights
    state_dict = torch.load(model_path)
    
    # Determine model architecture and number of classes
    if 'fc.weight' in state_dict:  # ResNet
        num_classes = state_dict['fc.weight'].size(0)
        model_name = 'resnet50'  # Assume ResNet50 by default
    elif 'classifier.1.weight' in state_dict:  # MobileNet or EfficientNet
        num_classes = state_dict['classifier.1.weight'].size(0)
        model_name = 'mobilenet' if not any('blocks' in k for k in state_dict.keys()) else 'efficientnet'
    else:
        raise ValueError("Cannot determine model architecture from state dict")
    
    # Create a new model with random initialization
    initial_model, _, _ = load_baseline_model(model_name, num_classes)
    
    # Save initial weights
    save_initial_weights(initial_model, initial_weights_path)
    
    # Find the lottery ticket
    winning_ticket, metrics = find_lottery_ticket(
        model_path=model_path,
        initial_weights_path=initial_weights_path,
        train_loader=train_loader,
        val_loader=val_loader,
        prune_percent=prune_percent,
        iterations=iterations,
        epochs_per_iteration=epochs_per_iteration,
        learning_rate=learning_rate
    )
    
    # Print metrics summary
    logger.info("Lottery Ticket Metrics Summary:")
    for m in metrics:
        logger.info(f"Iteration {m['iteration']}: Sparsity={m['sparsity']*100:.2f}%, "
                   f"Param Count={m['param_count']:,}, Val Acc={m['best_val_acc']:.2f}%")
    
    # Clean up temporary files
    if os.path.exists(initial_weights_path):
        os.remove(initial_weights_path)
    
    return winning_ticket


if __name__ == "__main__":
    # Test the lottery ticket implementation
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Lottery Ticket Hypothesis implementation")
    
    # This would require actual data loaders and models to run a real test
    # For demonstration purposes only
    logger.info("To use this module, import it in the main pipeline") 